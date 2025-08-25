
import numpy as np

class IterativeClosestPoint:

    def __init__(self, Q : np.ndarray, P : np.ndarray):
        
        self.Q = Q

        self.P = P

        self.C = self.Q.copy()

        self.S = self.P.copy()

        self.N, self.D = self.Q.shape

        self.Lambda = np.zeros(shape = (self.N, self.N))

        self.delta = np.zeros(shape = self.N)

        self.R = np.zeros(shape = (self.D, self.D))

        self.mu = np.zeros(shape = self.D)

        self.epsilon = 0

        self.gamma = 0

    def update_Lambda(self) -> None:

        self.Lambda = np.expand_dims(self.Q, 1) - np.expand_dims(self.S, 0)

        self.Lambda = np.linalg.norm(self.Lambda, ord = 2, axis = 2)

    def update_delta(self) -> None:

        self.delta = np.argmin(self.Lambda, axis = 0)

    def update_C(self) -> None:

        self.C = self.Q[self.delta]

    def update_R(self) -> None:

        self.R = (self.C - self.C.mean(axis = 0)).T @ (self.S - self.S.mean(axis = 0))

        U, S, V = np.linalg.svd(self.R, full_matrices = True)

        self.R = (U @ V).T

    def update_mu(self) -> None:

        self.mu = self.S.mean(axis = 0) - self.R @ self.C.mean(axis = 0)

    def update_S(self) -> None:

        self.S = (self.S @ self.R).T - np.expand_dims(self.mu, axis = 1)

        self.S = self.S.T

    def update_epsilon(self) -> None:

        self.epsilon = np.diag(self.Lambda[self.delta]).mean()

    def update_parameters(self) -> None:

        self.update_Lambda()

        self.update_delta()

        self.update_C()

        self.update_R()

        self.update_mu()

        self.update_S()

        self.update_epsilon()

    def estimates_R(self) -> None:

        self.R = (self.C - self.C.mean(axis = 0)).T @ (self.P - self.P.mean(axis = 0))

        U, S, V = np.linalg.svd(self.R, full_matrices = True)

        self.R = (U @ V).T

    def estimates_mu(self) -> None:

        self.mu = self.P.mean(axis = 0) - self.R @ self.C.mean(axis = 0)

    def estimates_parameters(self) -> None:

        self.estimates_R()

        self.estimates_mu()

    def update_model(self, MAX : int = 1000, TOL : float = 1e-6) -> None:

        for i in range(MAX):

            self.gamma = self.epsilon

            self.update_parameters()

            self.gamma -= self.epsilon

            if np.abs(self.gamma) < TOL:

                break

        self.estimates_parameters()
