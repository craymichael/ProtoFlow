"""
https://github.com/ldeecke/gmm-torch
MIT License

Copyright (c) 2019 Lucas Deecke

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

------------------------------------------------------------------------------

GNU GPL v2.0
Copyright (c) 2024 Zachariah Carmichael, Timothy Redgrave, Daniel Gonzalez Cedre
ProtoFlow Project

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
"""
import math
import numpy as np
import torch
from torch.nn import functional as F
from torch import nn

from math import pi

from protoflow.layers import LogDropout


def calculate_matmul_n_times(n_components, mat_a, mat_b):
    res = torch.zeros(mat_a.shape, device=mat_a.device)

    for i in range(n_components):
        mat_a_i = mat_a[:, i, :, :].squeeze(-2)
        mat_b_i = mat_b[0, i, :, :].squeeze()
        res[:, i, :, :] = mat_a_i.mm(mat_b_i).unsqueeze(1)

    return res


def calculate_matmul(mat_a, mat_b):
    assert mat_a.shape[-2] == 1 and mat_b.shape[-1] == 1
    return torch.sum(mat_a.squeeze(-2) * mat_b.squeeze(-1), dim=2, keepdim=True)


class GaussianMixture(nn.Module):

    def __init__(self, n_components, n_features, covariance_type="full",
                 eps=1.e-6, init_params="kmeans", mu_init=None, var_init=None,
                 requires_grad=False, dropout_prob=None):

        super(GaussianMixture, self).__init__()

        self.n_components = n_components
        self.n_features = n_features

        self.mu_init = mu_init
        self.var_init = var_init
        self.eps = eps

        self.log_likelihood = -torch.inf

        self.covariance_type = covariance_type
        self.init_params = init_params

        self.requires_grad = requires_grad

        assert self.covariance_type in ["full", "diag"]
        assert self.init_params in ["kmeans", "random"]

        self.dropout = LogDropout(
            p=dropout_prob) if dropout_prob else nn.Identity()

        self._init_params()

    def _init_params(self):
        if self.mu_init is not None:
            assert self.mu_init.size() == (1, self.n_components,
                                           self.n_features), "Input mu_init does not have required tensor dimensions (1, %i, %i)" % (
                self.n_components, self.n_features)

            self.mu = nn.Parameter(self.mu_init,
                                   requires_grad=self.requires_grad)
        else:
            self.mu = nn.Parameter(
                torch.randn(1, self.n_components, self.n_features),
                requires_grad=self.requires_grad)

        if self.covariance_type == "diag":
            if self.var_init is not None:

                assert self.var_init.size() == (1, self.n_components,
                                                self.n_features), "Input var_init does not have required tensor dimensions (1, %i, %i)" % (
                    self.n_components, self.n_features)
                self.var = nn.Parameter(self.var_init,
                                        requires_grad=self.requires_grad)
            else:
                self.var = nn.Parameter(
                    torch.ones(1, self.n_components, self.n_features),
                    requires_grad=self.requires_grad)
        elif self.covariance_type == "full":
            if self.var_init is not None:

                assert self.var_init.size() == (
                    1, self.n_components, self.n_features,
                    self.n_features), "Input var_init does not have required tensor dimensions (1, %i, %i, %i)" % (
                    self.n_components, self.n_features, self.n_features)
                self.var = nn.Parameter(self.var_init,
                                        requires_grad=self.requires_grad)
            else:
                self.var = nn.Parameter(
                    torch.eye(self.n_features).reshape(1, 1, self.n_features,
                                                       self.n_features).repeat(
                        1, self.n_components, 1, 1),
                    requires_grad=self.requires_grad
                )

        self.pi = nn.Parameter(
            torch.Tensor(1, self.n_components, 1).fill_(
                1. / self.n_components),
            requires_grad=self.requires_grad)
        self.params_fitted = False

    def check_size(self, x):
        if len(x.size()) == 2:
            x = x.unsqueeze(1)

        return x

    def bic(self, x):

        x = self.check_size(x)
        n = x.shape[0]

        free_params = self.n_features * self.n_components + self.n_features + self.n_components - 1

        bic = -2. * self.__score(
            x, as_average=False).mean() * n + free_params * torch.log(n)

        return bic

    def fit(self, x, delta=1e-3, n_iter=100, warm_start=False):

        if not warm_start and self.params_fitted:
            self._init_params()

        x = self.check_size(x)

        if self.init_params == "kmeans" and self.mu_init is None:
            mu = self.get_kmeans_mu(x, n_centers=self.n_components)
            self.mu.data = mu

        i = 0
        j = torch.inf

        while (i <= n_iter) and (j >= delta):

            log_likelihood_old = self.log_likelihood
            mu_old = self.mu
            var_old = self.var

            self.__em(x)
            self.log_likelihood = self.__score(x)

            if torch.isinf(self.log_likelihood.abs()) or torch.isnan(
                    self.log_likelihood):
                device = self.mu.device

                self.__init__(self.n_components,
                              self.n_features,
                              covariance_type=self.covariance_type,
                              mu_init=self.mu_init,
                              var_init=self.var_init,
                              eps=self.eps)
                for p in self.parameters():
                    p.data = p.data.to(device)
                if self.init_params == "kmeans":
                    self.mu.data, = self.get_kmeans_mu(x,
                                                       n_centers=self.n_components)

            i += 1
            j = self.log_likelihood - log_likelihood_old

            if j <= delta:
                self.__update_mu(mu_old)
                self.__update_var(var_old)

        self.params_fitted = True

    def predict(self, x, probs=False):

        x = self.check_size(x)

        weighted_log_prob = self._estimate_log_prob(x) + torch.log(
            F.softmax(self.pi, dim=1))

        if probs:
            p_k = self.dropout(torch.exp(weighted_log_prob))
            return torch.squeeze(p_k / (p_k.sum(1, keepdim=True)), dim=2)
        else:
            return torch.squeeze(
                torch.max(weighted_log_prob, 1)[1].type(torch.LongTensor),
                dim=1)

    def predict_proba(self, x):

        return self.predict(x, probs=True)

    def log_prob_all(self, x):
        weighted_log_prob = self._estimate_log_prob(x) + torch.log(
            F.softmax(self.pi, dim=1))
        return self.dropout(weighted_log_prob.squeeze(dim=-1))

    def sample(self, n, y=None):

        if y is None:
            counts = torch.distributions.multinomial.Multinomial(
                total_count=n,
                probs=F.softmax(self.pi, dim=1).squeeze(dim=2).squeeze(dim=0)
            ).sample()
            y = torch.cat([
                torch.full([int(sample)], j, device=counts.device) for
                j, sample in enumerate(counts)
            ])
        else:
            if isinstance(y, int):

                counts = torch.tensor([0] * self.n_components)
                counts[y] = n
                y = torch.full((n,), y)
            else:
                raise NotImplementedError

        x = []
        for k in torch.arange(
                self.n_components, device=counts.device
        )[counts > 0]:
            if self.covariance_type == "diag":
                x_k = self.mu[0, k] + torch.randn(
                    int(counts[k]), self.n_features, device=self.pi.device
                ) * torch.sqrt(self.var[0, k])
            elif self.covariance_type == "full":
                d_k = torch.distributions.multivariate_normal.MultivariateNormal(
                    self.mu[0, k], self.var[0, k])
                x_k = torch.stack([d_k.sample() for _ in range(int(counts[k]))])
            else:
                raise NotImplementedError
            x.append(x_k)

        x = torch.cat(x, dim=0)
        return x, y

    def score_samples(self, x):

        x = self.check_size(x)

        score = self.__score(x, as_average=False)
        return score

    def _estimate_log_prob(self, x):

        x = self.check_size(x)

        if self.covariance_type == "full":
            mu = self.mu
            var = self.var

            precision = torch.inverse(var)
            d = x.shape[-1]

            log_2pi = d * math.log(2. * pi)

            log_det = self._calculate_log_det(precision)

            x_mu_T = (x - mu).unsqueeze(-2)
            x_mu = (x - mu).unsqueeze(-1)

            x_mu_T_precision = calculate_matmul_n_times(self.n_components,
                                                        x_mu_T, precision)
            x_mu_T_precision_x_mu = calculate_matmul(x_mu_T_precision, x_mu)

            return -.5 * (log_2pi - log_det + x_mu_T_precision_x_mu)

        elif self.covariance_type == "diag":
            mu = self.mu
            prec = torch.rsqrt(self.var)

            log_p = torch.sum((mu * mu + x * x - 2 * x * mu) * prec, dim=2,
                              keepdim=True)
            log_det = torch.sum(torch.log(prec), dim=2, keepdim=True)

            return -.5 * (self.n_features * math.log(2. * pi) + log_p - log_det)

    def _calculate_log_det(self, var):

        log_det = torch.empty(size=(self.n_components,), device=var.device)

        for k in range(self.n_components):
            log_det[k] = 2 * torch.log(
                torch.diagonal(torch.linalg.cholesky(var[0, k]))).sum()

        return log_det.unsqueeze(-1)

    def _e_step(self, x):

        x = self.check_size(x)

        weighted_log_prob = self._estimate_log_prob(x) + torch.log(
            F.softmax(self.pi, dim=1))

        log_prob_norm = torch.logsumexp(weighted_log_prob, dim=1, keepdim=True)
        log_resp = weighted_log_prob - log_prob_norm

        return torch.mean(log_prob_norm), log_resp

    def _m_step(self, x, log_resp):

        x = self.check_size(x)

        resp = torch.exp(log_resp)

        pi = torch.sum(resp, dim=0, keepdim=True) + self.eps
        mu = torch.sum(resp * x, dim=0, keepdim=True) / pi

        if self.covariance_type == "full":
            eps = (torch.eye(self.n_features, device=x.device) * self.eps)
            var = torch.sum((x - mu).unsqueeze(-1).matmul(
                (x - mu).unsqueeze(-2)) * resp.unsqueeze(-1), dim=0,
                            keepdim=True) / torch.sum(resp, dim=0,
                                                      keepdim=True).unsqueeze(
                -1) + eps

        elif self.covariance_type == "diag":
            x2 = (resp * x * x).sum(0, keepdim=True) / pi
            mu2 = mu * mu
            xmu = (resp * mu * x).sum(0, keepdim=True) / pi
            var = x2 - 2 * xmu + mu2 + self.eps

        pi = pi / x.shape[0]

        return pi, mu, var

    def __em(self, x):

        _, log_resp = self._e_step(x)
        pi, mu, var = self._m_step(x, log_resp)

        self.__update_pi(pi)
        self.__update_mu(mu)
        self.__update_var(var)

    def __score(self, x, as_average=True):

        weighted_log_prob = self.dropout(self._estimate_log_prob(x) + torch.log(
            F.softmax(self.pi, dim=1)))
        per_sample_score = torch.logsumexp(weighted_log_prob, dim=1)

        if as_average:
            return per_sample_score.mean()
        else:
            return torch.squeeze(per_sample_score, dim=-1)

    def __update_mu(self, mu):

        assert mu.size() in [(self.n_components, self.n_features), (
            1, self.n_components,
            self.n_features)], "Input mu does not have required tensor dimensions (%i, %i) or (1, %i, %i)" % (
            self.n_components, self.n_features, self.n_components,
            self.n_features)

        if mu.size() == (self.n_components, self.n_features):
            self.mu = mu.unsqueeze(0)
        elif mu.size() == (1, self.n_components, self.n_features):
            self.mu.data = mu

    def __update_var(self, var):

        if self.covariance_type == "full":
            assert var.size() in [
                (self.n_components, self.n_features, self.n_features), (
                    1, self.n_components, self.n_features,
                    self.n_features)], "Input var does not have required tensor dimensions (%i, %i, %i) or (1, %i, %i, %i)" % (
                self.n_components, self.n_features, self.n_features,
                self.n_components, self.n_features, self.n_features)

            if var.size() == (
                    self.n_components, self.n_features, self.n_features):
                self.var = var.unsqueeze(0)
            elif var.size() == (
                    1, self.n_components, self.n_features, self.n_features):
                self.var.data = var

        elif self.covariance_type == "diag":
            assert var.size() in [(self.n_components, self.n_features), (
                1, self.n_components,
                self.n_features)], "Input var does not have required tensor dimensions (%i, %i) or (1, %i, %i)" % (
                self.n_components, self.n_features, self.n_components,
                self.n_features)

            if var.size() == (self.n_components, self.n_features):
                self.var = var.unsqueeze(0)
            elif var.size() == (1, self.n_components, self.n_features):
                self.var.data = var

    def __update_pi(self, pi):

        assert pi.size() in [(1, self.n_components,
                              1)], "Input pi does not have required tensor dimensions (%i, %i, %i)" % (
            1, self.n_components, 1)

        self.pi.data = pi

    def get_kmeans_mu(self, x, n_centers, init_times=50, min_delta=1e-3):

        if len(x.size()) == 3:
            x = x.squeeze(1)
        x_min, x_max = x.min(), x.max()
        x = (x - x_min) / (x_max - x_min)

        min_cost = torch.inf

        for i in range(init_times):
            tmp_center = x[
                np.random.choice(np.arange(x.shape[0]), size=n_centers,
                                 replace=False), ...]
            l2_dis = torch.norm(
                (x.unsqueeze(1).repeat(1, n_centers, 1) - tmp_center), p=2,
                dim=2)
            l2_cls = torch.argmin(l2_dis, dim=1)

            cost = 0
            for c in range(n_centers):
                cost += torch.norm(x[l2_cls == c] - tmp_center[c], p=2,
                                   dim=1).mean()

            if cost < min_cost:
                min_cost = cost
                center = tmp_center

        delta = torch.inf

        while delta > min_delta:
            l2_dis = torch.norm(
                (x.unsqueeze(1).repeat(1, n_centers, 1) - center), p=2, dim=2)
            l2_cls = torch.argmin(l2_dis, dim=1)
            center_old = center.clone()

            for c in range(n_centers):
                center[c] = x[l2_cls == c].mean(dim=0)

            delta = torch.norm((center_old - center), dim=1).max()

        return (center.unsqueeze(0) * (x_max - x_min) + x_min)


from torch import distributions


class SSLGaussMixture(nn.Module):

    def __init__(self, means, inv_cov_stds=None, device=None,
                 dropout_prob=None):
        super().__init__()
        self.covariance_type = 'full'
        self.n_components, self.d = means.shape
        self.mu = nn.Parameter(means)

        if inv_cov_stds is None:
            self.inv_cov_stds = math.log(math.exp(1.0) - 1.0) * torch.ones(
                (len(means)), device=device)
        else:
            self.inv_cov_stds = inv_cov_stds
        self.inv_cov_stds = nn.Parameter(self.inv_cov_stds)

        self.weights = nn.Parameter(torch.ones((len(means)), device=device))
        self.dropout = LogDropout(
            p=dropout_prob) if dropout_prob else nn.Identity()

    @property
    def var(self):
        return F.softplus(self.inv_cov_stds) ** 2

    @property
    def gaussians(self):
        gaussians = [
            distributions.MultivariateNormal(mean, F.softplus(
                inv_std) ** 2 * torch.eye(self.d, device=mean.device))
            for mean, inv_std in zip(self.mu, self.inv_cov_stds)
        ]
        return gaussians

    def sample(self, sample_shape, gaussian_id=None):
        if gaussian_id is not None:
            g = self.gaussians[gaussian_id]
            samples = g.sample(sample_shape)
        else:
            n_samples = sample_shape[0]
            idx = np.random.choice(self.n_components, size=(n_samples, 1),
                                   p=F.softmax(self.weights, dim=0))
            all_samples = [g.sample(sample_shape) for g in self.gaussians]
            samples = all_samples[0]
            for i in range(self.n_components):
                mask = np.where(idx == i)[0]
                samples[mask] = all_samples[i][mask]
        return samples

    def log_prob(self, x, y=None, label_weight=1.):
        all_log_probs = torch.cat(
            [g.log_prob(x)[:, None] for g in self.gaussians], dim=1)
        mixture_log_probs = torch.logsumexp(
            self.dropout(
                all_log_probs + torch.log(F.softmax(self.weights, dim=0))),
            dim=1)
        if y is not None:
            log_probs = torch.zeros_like(mixture_log_probs)
            mask = (y == -1)
            log_probs[mask] = mixture_log_probs[mask]
            for i in range(self.n_components):
                mask = (y == i)
                log_probs[mask] = self.dropout(
                    all_log_probs[:, i][mask] * label_weight)
            return log_probs
        else:
            return mixture_log_probs

    def log_prob_all(self, x):
        return self.class_logits(x)

    def class_logits(self, x):
        log_probs = torch.cat([g.log_prob(x)[:, None] for g in self.gaussians],
                              dim=1)
        log_probs_weighted = log_probs + torch.log(
            F.softmax(self.weights, dim=0))
        return self.dropout(log_probs_weighted)

    def classify(self, x):
        log_probs = self.class_logits(x)
        return torch.argmax(log_probs, dim=1)

    def class_probs(self, x):
        log_probs = self.class_logits(x)
        return F.softmax(log_probs, dim=1)

    def score_samples(self, x):

        return self.log_prob(x)


from denseflow.distributions.normal import ConvNormal2d


class GaussianMixtureConv2d(nn.Module):
    def __init__(self, features_shape, n_components, dropout_prob=None):
        super().__init__()
        self.features_shape = features_shape
        self.n_components = n_components
        self.gaussians = nn.ModuleList([
            ConvNormal2d(shape=features_shape)
            for _ in range(n_components)
        ])
        self.pi = nn.Parameter(
            torch.Tensor(self.n_components).fill_(
                1. / self.n_components)
        )
        self.dropout = LogDropout(
            p=dropout_prob) if dropout_prob else nn.Identity()

    def log_prob(self, x, y=None, label_weight=1.):
        x = x.reshape(-1, *self.features_shape)
        all_log_probs = torch.cat(
            [g.log_prob(x)[:, None] for g in self.gaussians], dim=1)
        mixture_log_probs = torch.logsumexp(self.dropout(
            all_log_probs + torch.log(F.softmax(self.pi, dim=0))), dim=1)
        if y is not None:
            log_probs = torch.zeros_like(mixture_log_probs)
            mask = (y == -1)
            log_probs[mask] = mixture_log_probs[mask]
            for i in range(self.n_components):
                mask = (y == i)
                log_probs[mask] = self.dropout(
                    all_log_probs[:, i][mask] * label_weight)
            return log_probs
        else:
            return mixture_log_probs

    def log_prob_all(self, x):
        all_log_probs = torch.cat(
            [g.log_prob(x)[:, None] for g in self.gaussians], dim=1)
        return self.dropout(
            all_log_probs + torch.log(F.softmax(self.pi, dim=0)))

    def score_samples(self, x):
        return self.log_prob(x)

    def sample(self, sample_shape, gaussian_id=None):
        if gaussian_id is not None:
            g = self.gaussians[gaussian_id]
            samples = g.sample(sample_shape)
        else:
            n_samples = sample_shape[0]
            idx = np.random.choice(self.n_components, size=(n_samples, 1),
                                   p=F.softmax(self.pi, dim=0))
            all_samples = [g.sample(sample_shape) for g in self.gaussians]
            samples = all_samples[0]
            for i in range(self.n_components):
                mask = np.where(idx == i)[0]
                samples[mask] = all_samples[i][mask]
        return samples
