#!/usr/bin/env python3
import jax.numpy as jnp
import jax.scipy as jsp

from mogpjaxe.experts import SVGPExperts

# def _log_prob(self, x):
#     x = tf.convert_to_tensor(x, name='x')
#     distribution_log_probs = [d.log_prob(x) for d in self.components]
#     cat_log_probs = self._cat_probs(log_probs=True)
#     final_log_probs = [
#         cat_lp + d_lp
#         for (cat_lp, d_lp) in zip(cat_log_probs, distribution_log_probs)
#     ]
#     concat_log_probs = tf.stack(final_log_probs, 0)
#     log_sum_exp = tf.reduce_logsumexp(concat_log_probs, axis=[0])
#     return log_sum_exp
# def _log_prob(self, k):
# logits = self.logits_parameter()
# if self.validate_args:
#   k = distribution_util.embed_check_integer_casting_closed(
#       k, target_dtype=self.dtype)
# k, logits = _broadcast_cat_event_and_params(
#     k, logits, base_dtype=dtype_util.base_dtype(self.dtype))
# return -tf.nn.sparse_softmax_cross_entropy_with_logits(
# cost = -np.sum(
# np.where(labels == 0, np.zeros_like(labels),labels * (logits - reduce_logsumexp(logits, axis=-1, keepdims=True))),axis=-1)
#     labels=k, logits=logits)
# @validate_sample
#     def log_prob(self, value):
#         M = _batch_mahalanobis(self.scale_tril, value - self.loc)
#         half_log_det = jnp.log(jnp.diagonal(self.scale_tril, axis1=-2, axis2=-1)).sum(
#             -1
#         )
#         normalize_term = half_log_det + 0.5 * self.scale_tril.shape[-1] * jnp.log(
#             2 * jnp.pi
#         )
#         return -0.5 * M - normalize_term


def mahalanobis(scale_tril, diff):
    # no need to use the below optimization procedure
    # solve_bL_bx = solve_triangular(bL, bx[..., None], lower=True).squeeze(-1)
    solve_bL_bx = jsp.linalg.solve_triangular(
        scale_til, diff[..., None], lower=True
    ).squeeze(-1)
    return jnp.sum(jnp.square(solve_bL_bx), -1)


def mvn_log_prob(mean, cov, value):
    scale_tril = jsp.linalg.cholesky(cov, lower=True)
    M = mahalanobis(scale_tril, value - mean)
    half_log_det = jnp.log(jnp.diagonal(scale_tril, axis1=-2, axis2=-1)).sum(-1)
    normalize_term = half_log_det + 0.5 * scale_tril.shape[-1] * jnp.log(2 * jnp.pi)
    return -0.5 * M - normalize_term


def gaussian_mixture_log_prob(means, covs, mixing_probs, value):
    cost = -jnp.sum(
        jnp.where(
            labels == 0,
            jnp.zeros_like(labels),
            labels * (logits - reduce_logsumexp(logits, axis=-1, keepdims=True)),
        ),
        axis=-1,
    )
    return 0


def lower_bound_further(
    params: dict, batch, gating_network, experts: SVGPExperts, num_var_samples: int = 1
):
    X, Y = data
    kl_gating = gating_network.prior_kls(params["gating_network"])
    kl_experts = experts.prior_kls(params["experts"])

    def variational_expectation_gibbs(x, y):
        mixing_probs_samples = gating_network.predict_mixing_probs(
            params["gating_network"], x, num_samples=num_var_samples
        )
        f_samples = experts.predict_fs_samples(
            params["experts"], x, num_samples=num_var_samples, full_cov=False
        )
        covs = experts.noise_variances

        def gaussian_mixture_log_prob_wrapper(means, mixing_probs):
            return gaussian_mixture_log_prob(means, covs, mixing_probs, value=y)

        var_exp_samples = jax.vmap(
            gaussian_mixture_log_prob_wrapper, f_means_samples, mixing_probs_samples
        )
        var_exp = jnp.sum(var_exp_samples, axis=0) / num_var_samples
        print("variational_expectation after averaging gibbs samples")
        print(var_exp.shape)
        return

    batched_var_exp = jax.vmap(variational_expectation, X, Y)
    sum_var_exp = jnp.sum(batched_var_exp, axis=0)
    print("sum_var_exp after sum over data mini batches")
    print(sum_var_exp.shape)

    if num_data is not None:
        scale = num_data / X.shape[0]
    else:
        scale = 1.0

    return sum_var_exp * scale - kl_gating - kl_experts


def lower_bound_1(params, batch, gating_network, experts, num_inducing_samples=1):
    X, Y = data
    kl_gating = gating_network.prior_kls(params["gating_network"])
    kl_experts = experts.prior_kls(params["experts"])

    # q_mu = params["gating_network"]["gating_functions"]["q_mu"]
    # q_sqrt = params["gating_network"]["gating_functions"]["q_sqrt"]
    # q_samples = sample_mvn(q_mu, q_sqrt)

    # mixing_probs = gating_network.gating_functions.predict_predict_mixing_probs(
    #     params["gating_network"], x
    # )

    def variational_expectation_gibbs(x, y):
        mixing_probs_samples = gating_network.predict_mixing_probs(
            params["gating_network"], x, num_inducing_samples=num_inducing_samples
        )
        f_means_samples, f_vars_samples = experts.predict_fs(
            params["experts"],
            x,
            num_inducing_samples=num_inducing_samples,
            full_cov=False,
        )
        var_samples = experts.noise_variances + f_vars_samples

        def gaussian_mixture_log_prob_wrapper(means, covs, mixing_probs):
            return gaussian_mixture_log_prob(means, covs, mixing_probs, value=y)

        var_exp_samples = jax.vmap(
            gaussian_mixture_log_prob_wrapper,
            f_means_samples,
            var_samples,
            mixing_probs_samples,
        )
        # average samples (gibbs) TODO have I averaged gibbs samples correctly???
        var_exp = jnp.sum(var_exp_samples, axis=0) / num_inducing_samples
        print("variational_expectation after averaging gibbs samples")
        print(var_exp.shape)
        return var_exp

    batched_var_exp = jax.vmap(variational_expectation_gibbs, X, Y)
    sum_var_exp = jnp.sum(batched_var_exp, axis=0)
    print("sum_var_exp after sum over data mini batches")
    print(sum_var_exp.shape)

    if num_data is not None:
        scale = num_data / X.shape[0]
    else:
        scale = 1.0

    return sum_var_exp * scale - kl_gating - kl_experts


def lower_bound_further(self, data: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
    """Lower bound to the log-marginal likelihood (ELBO).

    Looser bound than lower_bound_1 but analytically marginalises
    the inducing variables $q(\hat{f}, \hat{h})$. Replaces M-dimensional
    approx integrals with 1-dimensional approx integrals.

    This bound assumes each output dimension is independent.

    :param data: data tuple (X, Y) with inputs [num_data, input_dim]
                    and outputs [num_data, ouput_dim])
    :returns: loss - a Tensor with shape ()
    """
    with tf.name_scope("ELBO") as scope:
        X, Y = data
        num_test = X.shape[0]

        # kl_gating = self.gating_network.prior_kl()
        # kls_gatings = self.gating_network.prior_kls()
        kl_gating = tf.reduce_sum(self.gating_network.prior_kls())
        kl_experts = tf.reduce_sum(self.experts.prior_kls())

        mixing_probs = self.predict_mixing_probs(X)
        print("Mixing probs")
        print(mixing_probs.shape)

        # num_samples = 5
        num_samples = 1
        # TODO move this to a variational_expectation() method
        noise_variances = self.experts.noise_variances()
        fmeans, fvars = self.experts.predict_fs(X, full_cov=False)
        f_dist = tfd.Normal(loc=fmeans, scale=fvars)
        f_dist_samples = f_dist.sample(num_samples)

        components = []
        for expert_k in range(self.num_experts):
            component = tfd.Normal(
                loc=f_dist_samples[..., expert_k], scale=noise_variances[expert_k]
            )
            components.append(component)
        mixing_probs_broadcast = tf.expand_dims(mixing_probs, 0)
        mixing_probs_broadcast = tf.broadcast_to(
            mixing_probs_broadcast, f_dist_samples.shape
        )
        categorical = tfd.Categorical(probs=mixing_probs_broadcast)
        mixture = tfd.Mixture(cat=categorical, components=components)
        variational_expectation = mixture.log_prob(Y)
        print("variational_expectation")
        print(variational_expectation.shape)

        # sum over output dimensions (assumed independent)
        variational_expectation = tf.reduce_sum(variational_expectation, -1)
        print("variational_expectation after sum over output dims")
        print(variational_expectation.shape)

        # average samples (gibbs)
        # TODO have I average gibbs samples correctly???
        approx_variational_expectation = (
            tf.reduce_sum(variational_expectation, axis=0) / num_samples
        )
        print("variational_expectation after averaging gibbs samples")
        print(approx_variational_expectation.shape)
        sum_variational_expectation = tf.reduce_sum(
            approx_variational_expectation, axis=0
        )
        print("variational_expectation after sum over data mini batches")
        print(sum_variational_expectation.shape)

        if self.num_data is not None:
            num_data = tf.cast(self.num_data, default_float())
            minibatch_size = tf.cast(tf.shape(X)[0], default_float())
            scale = num_data / minibatch_size
        else:
            scale = tf.cast(1.0, default_float())
        return sum_variational_expectation * scale - kl_gating - kl_experts


def lower_bound_1(self, data: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
    """Lower bound to the log-marginal likelihood (ELBO).

    Tighter bound than lower_bound_further but requires an M dimensional
    expectation over the inducing variables $q(\hat{f}, \hat{h})$
    to be approximated (with Gibbs sampling).

    :param data: data tuple (X, Y) with inputs [num_data, input_dim]
                    and outputs [num_data, ouput_dim])
    :returns: loss - a Tensor with shape ()
    """
    with tf.name_scope("ELBO") as scope:
        X, Y = data
        num_test = X.shape[0]

        # kl_gating = self.gating_network.prior_kl()
        # kls_gatings = self.gating_network.prior_kls()
        kl_gating = tf.reduce_sum(self.gating_network.prior_kls())
        kl_experts = tf.reduce_sum(self.experts.prior_kls())

        with tf.name_scope("predict_mixing_probs") as scope:
            mixing_probs = self.predict_mixing_probs(
                X, num_inducing_samples=self.num_inducing_samples
            )
        # TODO move this reshape into gating function
        # mixing_probs = tf.reshape(
        #     mixing_probs,
        #     [self.num_inducing_samples, num_test, self.num_experts])
        print("Mixing probs")
        print(mixing_probs.shape)

        with tf.name_scope("predict_experts_prob") as scope:
            batched_dists = self.predict_experts_dists(
                X, num_inducing_samples=self.num_inducing_samples
            )

            Y = tf.expand_dims(Y, 0)
            Y = tf.expand_dims(Y, -1)
            expected_experts = batched_dists.prob(Y)
            print("Expected experts")
            print(expected_experts.shape)
            # TODO is it correct to sum over output dimension?
            # sum over output_dim
            expected_experts = tf.reduce_prod(expected_experts, -2)
            print("Experts after product over output dims")
            # print(expected_experts.shape)
            expected_experts = tf.expand_dims(expected_experts, -2)
            print(expected_experts.shape)

        shape_constraints = [
            (
                expected_experts,
                ["num_inducing_samples", "num_data", "1", "num_experts"],
            ),
            (
                mixing_probs,
                ["num_inducing_samples", "num_data", "1", "num_experts"],
            ),
        ]
        tf.debugging.assert_shapes(
            shape_constraints,
            message="Gating network and experts dimensions do not match",
        )
        with tf.name_scope("marginalise_indicator_variable") as scope:
            weighted_sum_over_indicator = tf.matmul(
                expected_experts, mixing_probs, transpose_b=True
            )

            # remove last two dims as artifacts of marginalising indicator
            weighted_sum_over_indicator = weighted_sum_over_indicator[:, :, 0, 0]
        print("Marginalised indicator variable")
        print(weighted_sum_over_indicator.shape)

        # TODO where should output dimension be reduced?
        # weighted_sum_over_indicator = tf.reduce_sum(
        #     weighted_sum_over_indicator, (-2, -1))
        # weighted_sum_over_indicator = tf.reduce_sum(
        #     weighted_sum_over_indicator, (-2, -1))
        # print('Reduce sum over output dimension')
        # print(weighted_sum_over_indicator.shape)

        # TODO correct num samples for K experts. This assumes 2 experts
        num_samples = self.num_inducing_samples ** (self.num_experts + 1)
        var_exp = (
            1
            / num_samples
            * tf.reduce_sum(tf.math.log(weighted_sum_over_indicator), axis=0)
        )
        print("Averaged inducing samples")
        print(var_exp.shape)
        # # TODO where should output dimension be reduced?
        # var_exp = tf.linalg.diag_part(var_exp)
        # print('Ignore covariance in output dimension')
        # print(var_exp.shape)
        var_exp = tf.reduce_sum(var_exp)
        print("Reduced sum over mini batch")
        print(var_exp.shape)

        # var_exp = tf.reduce_sum(var_exp)
        # print('Reduce sum over output_dim to get loss')
        # print(var_exp.shape)

        if self.num_data is not None:
            num_data = tf.cast(self.num_data, default_float())
            minibatch_size = tf.cast(tf.shape(X)[0], default_float())
            scale = num_data / minibatch_size
        else:
            scale = tf.cast(1.0, default_float())

        return var_exp * scale - kl_gating - kl_experts
