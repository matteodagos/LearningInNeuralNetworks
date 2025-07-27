import numpy as np

def ReLU(x : np.ndarray, net : 'Visual_network') -> np.ndarray:
    """
    ReLU activation function
    :param x: input value
    :param net: unused but required for compatibility with other transfer functions
    :return: ReLU of x
    """
    return np.maximum(0, x)

def BCM(y : np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
        Non linearity function of the Hebbian learning rule
        :param y: output neuron value (after its transfer function), shape (self.n_out,)
        :param theta: Estimation of the mean square output of the neuron. Same shape as u.
    """
    return y * (y - theta)

def non_competitive_update(
        net: 'Visual_network',
        x: np.ndarray,
        gamma: float,
        theta: np.ndarray,
        tau: float = None,
) -> None:
    """
    Update the weights of the network using the non competitive update rule.
    :param net: the network to be updated
    :param x: input vector of shape (n_in,)
    :param gamma: learning rate
    :param theta: parameter of the non linearity function. Here the estimation of the mean square output
    :param tau: if specified, use it to update theta. Else, theta is not updated.
    :return: nothing, update the weights of the network in place
    """

    y = net.forward(x)
    delta_w = gamma * net.f(y, theta)[:, np.newaxis] * x

    if tau:
        theta += 1/tau * (y**2 - theta)
    new_w = net.weights + delta_w
    # row wise l2 normalization
    net.weights = new_w / np.linalg.norm(new_w, axis=1, keepdims=True)


def create_V(n_out, seed=0):
    v = np.abs(np.random.RandomState(seed).randn(n_out, n_out))
    np.fill_diagonal(v, 0)
    return v


def competitive_transfer_function(wx: np.ndarray, net: 'Visual_network') -> (np.ndarray, np.ndarray):
    """
    Compute the competitive transfer function for the given inputs.
    :param net:
    :param wx: the neuron input potentials, shape (n_out,)
    :return: the neurons outputs y and the neuron output potentials u (shape (n_out,))
    """
    V = net.transfer_function_params['V']
    dt = net.transfer_function_params['dt']
    n_steps = net.transfer_function_params['n_steps']

    u = wx  # TODO I dont have any justification (beside gpt) to find a good initalization

    for _ in range(n_steps):
        relu_u = np.maximum(0, u)
        du_dt = -u + wx - V @ relu_u
        u = u + dt * du_dt

    return np.maximum(0, u), u


def competitive_non_linearity_W(u: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
    Compute the competitive non-linearity for learning the forward weights ,given inputs, using BCM rule
    :param u: neuron outputs potentials
    :param theta: output neurons estimated mean value
    :return: non linearity of the neuron outputs to be used in the learning rule
    """
    # TODO should we use np.maximum(0, u) here?
    return u * (u - theta)


def competitive_non_linearity_V(u: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """
    Compute the competitive non-linearity for learning the lateral weights, given inputs
    :param u: neurons outputs potentials
    :param phi: output neurons estimated mean value
    :return: non linearity of the neuron outputs to be used in the learning rule
    """
    return np.maximum(u, 0) - phi


def competitive_update(
        net: 'Visual_network',
        x: np.ndarray,
        gamma: float,
        theta: np.ndarray,
        phi: np.ndarray,
        v_list: list = None,
        tau : float = None,
        learn_theta: bool = False,
        learn_phi: bool = False
):
    if not tau and (learn_theta or learn_phi):
        raise ValueError("tau is not specified, theta and phi will not be updated even if learn_theta or learn_phi is True.")


    if v_list is not None:
        # if v_list is provided, we use it to keep track of the mean lateral weights
        v_list.append(net.transfer_function_params['V'].copy())

    v_non_linearity = net.transfer_function_params['v_non_linearity']

    y, u = net.forward(x)
    delta_w = gamma * net.rule_non_linearity(y, theta)[:, np.newaxis] * x
    delta_v = gamma * v_non_linearity(y, phi)[:, np.newaxis] * y

    if tau:

        if learn_theta: theta += 1/tau * (y**2 - theta)
        if learn_phi: phi += 1/tau * (y - phi)


    new_w = net.weights + delta_w
    new_v = net.transfer_function_params['V'] + delta_v

    new_w = new_w / np.linalg.norm(new_w, axis=1, keepdims=True)
    new_v = np.maximum(0, new_v)
    np.fill_diagonal(new_v, 0)

    net.weights = new_w
    net.transfer_function_params['V'] = new_v


class Visual_network:
    """
    :param n_in: number of input each n_out neuron receive
    :param n_out: number of out neurons
    :param seed: random seed for weight initialization. Default is 0
    :param rule_non_linearity: non linearity function of the Hebbian learning rule. Default is BCM
        Function of the form rule_non_linearity(y, theta),
        where y is the output neuron value (after its transfer function) and theta is a numpy array of parameters
    :param transfer_function: transfer function of the network. Default is ReLU.
        Need to be a function of the form transfer_function(wx, network),
        where wx is the input potential (weights @ network_input) and network is the network instance
    :param transfer_function_params: parameters used in the transfer function. Default is None.
    :param update_rule: update rule of the network. Default is non_competitive_update.
        Function of the form update_rule(network, x, gamma, theta, **kwargs),
        where network is the network instance, x is the input vector, gamma is the learning rate,
        theta is a vector of parameters for the non linearity function and kwargs are additional parameters.

    instantiate a network with a weight matrix of shape (n_out, n_in). network output is given by forward(x) = transfer_function(W @ x, network, x),
    the update rule is given by update(x, gamma, theta) = W += gamma * f(forward(x), theta) * x,
    where f is the non linearity function of the Hebbian learning rule.
    the update rule for theta is given by theta += 1/tau * (forward(x)**2 - theta) if tau is specified, else theta is not updated.

    """
    def __init__(
            self,
            n_in,
            n_out,
            transfer_function=ReLU,
            transfer_function_params=None,
            rule_non_linearity=BCM,
            update_rule=non_competitive_update,
            seed=0
    ):
        self.rule_non_linearity = rule_non_linearity
        self.transfer_function = transfer_function
        self.transfer_function_params = transfer_function_params
        self.update_rule = update_rule
        self.seed = seed
        self.n_in = n_in
        self.n_out = n_out
        self.weights = np.random.RandomState(self.seed).randn(n_out, n_in)

    def forward(self, x: np.ndarray, weights=None) -> np.ndarray:
        """
        returns state of self.n_out neurons given input x of shape (self.n_in,), ie apply weights and transfer function.
        Use the network weights by default but can optionally used a specified weight matrix.
        :param suppress_warning: to suppress the warning when using non network weights.
        :param weights: default is self.weights. numpy array of shape (self.n_out, self.n_in).
        :param x: numpy vector of shape (self.n_in,)
        :return: numpy vector of shape (self.n_out,)
        """
        if weights is None:
            weights = self.weights

        wx = np.dot(weights, x)
        return self.transfer_function(wx, self)

    def f(self, y : np.ndarray, theta: np.ndarray) -> np.ndarray:
        """
            rule non linearity function of the Hebbian learning rule
            :param y: output neuron value (after its transfer function), shape (self.n_out,)
            :param theta: additional parameters of the non linearity function. Here the estimation of the mean square output
            Return an array of same shape as y
        """
        return self.rule_non_linearity(
            y,
            theta
        )


    def update(self, x: np.ndarray, gamma: float, theta: np.ndarray,  **kwargs) -> None:
        """
        Uses the update rule defined in the constructor to update the weights of the network.
        :param x: input vector of shape (n_in,)
        :param gamma: learning rate
        :param theta: parameter of the non linearity function.
        :param kwargs: args to be passed to the update rule.
        """
        self.update_rule(self, x, gamma, theta, **kwargs)



