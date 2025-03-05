import torch as th
import itertools
from .model import RNN

def sampling_accuracy(rnn, n_test=2**20):
    x = th.randn(n_test, rnn.n, dtype=rnn.dtype, device=rnn.device)
    logits = rnn(x)
    pred = logits.argmax(dim=1)
    target = x.argsort(dim=1)[:, -2]
    
    pred = logits.argmax(dim=1)
    pred[(logits == 0).all(dim=1)] = -1

    return (pred == target).double().mean()

def condition_halfspace(mu: th.tensor, sigma: th.tensor, constraint: th.tensor) -> dict[str, th.tensor]:
    """
    Calculate the probability, conditional mean, and conditional covariance of the intersection of a single halfspace with a Gaussian. Used in covariance propagation. Works in O(n^2) time.
    mu is a batch of means, sigma is a batch of covariances, and constraint is a batch of halfspaces.
    Returns:
        dict of {mu, sigma, prob}
    """
    batch, n = mu.shape
    assert sigma.shape == (batch, n, n)
    assert constraint.shape == (batch, n,)
    assert (th.norm(constraint, dim=1) > 0).all(), "constraint must be a non-zero vector"

    a = constraint
    # Compute alpha = (a^T μ) / sqrt(a^T Σ a)
    # (batch, 1, n) @ (batch, n, n) @ (batch, n, 1) -> (batch, 1, 1)
    denom = th.sqrt(a.unsqueeze(1) @ sigma @ a.unsqueeze(-1)).squeeze(-1).squeeze(-1)
    # (batch, n) @ (batch, n) -> (batch,)
    alpha = (a * mu).sum(dim=1) / denom
    
    # Compute Mills ratio λ(α) = φ(α)/Φ(α)
    phi = th.exp(-0.5 * alpha**2) / (2 * th.pi) ** 0.5
    Phi = 0.5 * (1 + th.erf(alpha / (2**0.5)))
    mills_ratio = phi / Phi
    
    # Multiply by probability of this halfspace
    prob = Phi
    
    # Update mean: μ_new = μ + Σa * λ(α) / Φ(α)
    # (batch, n, n) @ (batch, n, 1) -> (batch, n, 1)
    sigma_a = sigma @ a.unsqueeze(-1)
    mu = mu + (sigma_a.squeeze(-1) * mills_ratio.unsqueeze(-1)) / denom.unsqueeze(-1)
    
    # Compute δ(α) = λ(α)(λ(α) + α)
    delta = mills_ratio * (mills_ratio + alpha)
    
    # Update covariance: Σ_new = Σ - Σa(a^TΣa)^(-1)a^TΣ * δ(α)
    # (batch, n, 1) @ (batch, 1, n) -> (batch, n, n)
    outer_prod = sigma_a @ (a.unsqueeze(1) @ sigma)
    sigma = sigma - (outer_prod * delta.unsqueeze(-1).unsqueeze(-1)) / (denom * denom).unsqueeze(-1).unsqueeze(-1)

    return dict(mu=mu, sigma=sigma, prob=prob)


def girard_solid_angle_3d(constraints: list[th.Tensor], EPS=1e-12):
    """
    Uses Girard's formula to calculate the solid angle of the spherical polygon formed by linear constraints. Works in O(len(constraints)^2) time, although more efficient algorithms are possible.
    constraints: a list of (tensors of length 3), each representing a linear constraint
    Requires that no constraint is a zero vector
        (a0 ... an) means a0*x0 + a1*x1 + a2*x2 >= 0
    """
    dtype = constraints[0].dtype
    device = constraints[0].device
    one = th.tensor(1.0, dtype=dtype, device=device, requires_grad=True)
    # remove duplicates
    con = []

    constraints_tensor = th.stack(constraints)
    constraints_tensor = constraints_tensor / th.norm(constraints_tensor, dim=1, keepdim=True)
    pairwise_dots = constraints_tensor @ constraints_tensor.T
    if (pairwise_dots < -1 + EPS).any():
        return one * 0.0
    
    pairwise_dots = pairwise_dots.triu(diagonal=1)
    con = list(constraints_tensor[(pairwise_dots < 1 - EPS).all(dim=1)])
    
    if len(con) == 0:
        return one * 4 * th.pi
    elif len(con) == 1:
        return one * 2 * th.pi
    elif len(con) == 2:
        # angle between two vectors
        return 2 * (th.pi - th.acos(con[0].dot(con[1])))
    

    # find the longest prefix that is coplanar
    coplanar = []
    for i in range(len(con)):
        c = con[i]
        if len(coplanar) <= 1:
            coplanar.append(c)
        else:
            v = th.linalg.cross(coplanar[-1], coplanar[-2])
            if abs(v.dot(c)) < EPS:
                coplanar.append(c)
            else:
                break

    # check if they all point in the same direction
    # find the two points that are furthest apart
    furthest = (0, 1)
    for i in range(len(coplanar)):
        for j in range(i+1, len(coplanar)):
            if (coplanar[i] - coplanar[j]).norm() > (coplanar[furthest[0]] - coplanar[furthest[1]]).norm():
                furthest = (i, j)
    mid = (coplanar[furthest[0]] + coplanar[furthest[1]])
    mid = mid / mid.norm()
    # make sure mid has positive dot product with all coplanar vectors
    if any(mid.dot(c) < EPS for c in coplanar):
        return one * 0.0
    
    con = [coplanar[furthest[0]], coplanar[furthest[1]]] + con[len(coplanar):]

    points = []
    # add first two points based on intersection of first two constraints
    p0 = th.linalg.cross(con[0], con[1])
    p0 = p0 / p0.norm()
    p1 = -p0
    q0 = th.linalg.cross(con[1], p0)
    q0 = q0 / q0.norm()
    q1 = th.linalg.cross(con[0], p1)
    q1 = q1 / q1.norm()
    points = [
        p0, q0, p1, q1
    ]

    for i in range(2, len(con)):
        # intersect with the new constraint
        new_points = []
        c = con[i]
        dots = [p.dot(c) for p in points]
        if all(d <= EPS for d in dots):
            # check if any points are on right side of c
            return one * 0.0
        if all(d >= -EPS for d in dots):
            continue
        
        start = 0
        while dots[start] < EPS:
            start += 1
        points = points[start:] + points[:start]
        dots = dots[start:] + dots[:start]
        points.append(points[0])
        dots.append(dots[0])

        new_points = []
        for i in range(len(points)-1):
            if dots[i] >= -EPS:
                new_points.append(points[i])
                if dots[i+1] < -EPS:
                    # crosses over
                    x, y = points[i], points[i+1]
                    z = (x * abs(dots[i+1]) + y * abs(dots[i]))
                    z = z / z.norm()
                    new_points.append(z)
            else:
                if dots[i+1] >= -EPS:
                    # crosses over
                    x, y = points[i], points[i+1]
                    z = (x * abs(dots[i+1]) + y * abs(dots[i]))
                    z = z / z.norm()
                    new_points.append(z)
        
        points = []
        for p in new_points:
            if len(points) == 0 or (p - points[-1]).norm() > EPS:
                points.append(p)
    
    # now, find angles between points
    assert len(points) >= 3

    assert all(abs(p.norm() - 1) < EPS for p in points)
    # make sure no two consecutive points are the same
    assert all((points[i] - points[(i+1)%len(points)]).norm() > EPS for i in range(len(points)))

    angles = []
    for i in range(len(points)):
        x, y, z = points[i], points[(i+1)%len(points)], points[(i+2)%len(points)]
        # get the spherical angle from xy to yz
        #first, find x' and z'; the projection of x and z onto the tangent plane at y
        x_tangent = x + y * (1-x.dot(y))
        z_tangent = z + y * (1-z.dot(y))
        a = x_tangent - y
        b = z_tangent - y
        a = a / a.norm()
        b = b / b.norm()

        dot = a.dot(b)
        if dot > -1:
            angles.append(th.acos(a.dot(b)))
        
    return sum(angles) - th.pi * (len(angles) - 2)

def exact_acc_rnn(rnn, EPS=1e-12):
    """
    Find the accuracy of the model by considering linear regions.
    Only works for n=3.
    """
    n = rnn.n
    assert n == 3, "only sequence length of 3 is supported"
    d = rnn.d

    all_regions = []

    # find the regions
    for mask in itertools.product([0, 1], repeat=n*d):
        # convert mask into tensor of shape (n, d)
        mask = th.tensor(mask, dtype=th.bool).reshape(n, d)
        x = th.zeros((d, n), dtype=rnn.dtype, device=rnn.device)

        constraints = []

        for i in range(n):
            x = rnn.Whh @ x
            x[:,i] += rnn.Whi[:,0]
            for j in range(d):
                if mask[i, j]: # relu is >= 0
                    constraints.append(x[j])
                else:
                    constraints.append(-x[j])
                    x[j] = 0
        
        logits = rnn.Woh @ x

        # Make a subregion for each possible (argmax, second-argmax)
        w = th.eye(n, dtype=rnn.dtype, device=rnn.device)
        for argmax in range(n):
            for second_argmax in range(n):
                if argmax != second_argmax:
                    extra_constraints = []

                    for i in range(n):
                        if i == second_argmax:
                            continue
                        # enforce the (everything else) <= second_argmax <= argmax
                        if i == argmax:
                            extra_constraints.append(w[i] - w[second_argmax])
                        else:
                            extra_constraints.append(w[second_argmax] - w[i])
                        # make sure the second argmax logit is the highest
                        extra_constraints.append(logits[second_argmax] - logits[i])

                    all_regions.append([c.clone() for c in constraints] + extra_constraints)

    ans = th.tensor(0.0, dtype=rnn.dtype, device=rnn.device)
    for region in all_regions:
        if any(c.norm() == 0 for c in region):
            continue
        s = girard_solid_angle_3d(region, EPS=EPS)
        ans += s
    return ans / (4 * th.pi)

def GMHP(rnn, C=None):
    """
    Gaussian Mixture Halfspace Pruning.
    Time complexity: O(C * nd * n^2) (where time to do intersections is n^2)
    If C is None, then O(n(n-1) * 2^{nd} * n^2)
    """
    device = rnn.device

    n = rnn.n
    d = rnn.d
    dtype = rnn.dtype

    # Initialize components for all possible (argmax, second_argmax) pairs
    n_initial = n * (n-1)  # number of (argmax, second_argmax) pairs
    mu = th.zeros(n_initial, n, dtype=dtype, device=device)
    sigma = th.eye(n, dtype=dtype, device=device).unsqueeze(0).repeat(n_initial, 1, 1)
    prob = th.ones(n_initial, dtype=dtype, device=device)
    H = th.zeros(n_initial, d, n, dtype=dtype, device=device)
    winners = th.zeros(n_initial, dtype=th.long, device=device)
    
    # Set up initial constraints
    w = th.eye(n, dtype=dtype, device=device)
    idx = 0
    for argmax in range(n):
        for second_argmax in range(n):
            if argmax == second_argmax: continue
            
            winners[idx] = second_argmax

            # Apply initial halfspace constraints
            for i in range(n):
                if i == second_argmax: continue
                constraint = w[i] - w[second_argmax] if i == argmax else w[second_argmax] - w[i]
                result = condition_halfspace(
                    mu[idx:idx+1], 
                    sigma[idx:idx+1], 
                    constraint.unsqueeze(0)
                )
                mu[idx] = result['mu'][0]
                sigma[idx] = result['sigma'][0]
                prob[idx] *= result['prob'][0]
            
            idx += 1

    # Process each layer
    for layer in range(n):
        # Update H
        H = rnn.Whh @ H
        H[:, :, layer] += rnn.Whi.squeeze(1)
        
        # Process ReLU for each neuron
        for i in range(d):
            # Find components that need splitting
            mask = th.norm(H[:, i], dim=1) > 0
            assert (mask.sum() == len(mu))
            
            # Since mask is all ones, we can simplify by just duplicating everything
            mu = th.cat([mu, mu], dim=0)
            sigma = th.cat([sigma, sigma], dim=0)
            prob = th.cat([prob, prob], dim=0)
            winners = th.cat([winners, winners], dim=0)
            
            n_split = len(mu) // 2
            constraint = H[:, i]
            
            # Positive halfspace
            pos_result = condition_halfspace(
                mu[:n_split],
                sigma[:n_split], 
                constraint
            )
            # Negative halfspace
            neg_result = condition_halfspace(
                mu[n_split:],
                sigma[n_split:],
                -constraint
            )
            
            # Zero out H for negative halfspace
            H = th.cat([H, H.clone()], dim=0)
            H[n_split:, i] = 0
            
            # Update results
            mu = th.cat([pos_result['mu'], neg_result['mu']], dim=0)
            sigma = th.cat([pos_result['sigma'], neg_result['sigma']], dim=0)
            prob = th.cat([prob[:n_split] * pos_result['prob'],
                          prob[n_split:] * neg_result['prob']], dim=0)

            if C is not None and len(prob) > C:
                # Keep top-C components by probability
                top_C = th.topk(prob, min(C, len(prob)), dim=0)
                indices = top_C.indices
                mu = mu[indices]
                sigma = sigma[indices]
                prob = prob[indices]
                H = H[indices]
                winners = winners[indices]

    # Final logit constraints
    logits = rnn.Woh @ H
    normalization = prob.sum()
    
    # Apply final constraints for each component
    for i in range(n):
        mask = winners != i
        if mask.any():
            constraint_mask = (th.norm(logits[th.arange(len(winners)), winners] - logits[th.arange(len(winners)), i], dim=1) > 0) & mask
            
            if constraint_mask.any():
                constraint = logits[constraint_mask, winners[constraint_mask]] - logits[constraint_mask, i]
                result = condition_halfspace(
                    mu[constraint_mask],
                    sigma[constraint_mask],
                    constraint
                )
                mu[constraint_mask] = result['mu']
                sigma[constraint_mask] = result['sigma']
                prob[constraint_mask] *= result['prob']

    return prob.sum() / normalization