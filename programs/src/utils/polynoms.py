import numpy as np

class Polynom:
    """Class for Polynoms.
    For simplicity here: deg(P=0) = deg(P=a, a≠0) = 0
    """
    def __init__(self, arg):
        if isinstance(arg, int):
            assert arg >= 0, f"Expected non-negative degree, got {arg}."
            self.deg = arg
            self.coeffs = np.zeros(arg + 1)
        elif isinstance(arg, np.ndarray):
            assert arg.ndim == 1, "Coefficients must be a 1D array."
            self.coeffs = arg.astype(float)
            self.deg = len(self.coeffs) - 1
        else:
            raise TypeError("Polynom constructor expects an int or a 1D numpy array.")

    def __call__(self, x: float):
        result = 0.0
        power = 1.0
        for c in self.coeffs:
            result += c * power
            power *= x
        return result

    def __str__(self):
        return f"<Polynom deg={self.deg}, coeffs={self.coeffs}>"

    def set_coeff(self, id_coeff:int, coeff:float):
        assert id_coeff<=self.deg and id_coeff>=0, ValueError(
            f"Expected a coefficient the polynom has, got {id_coeff}.")
        self.coeffs[id_coeff] = coeff
    
    def get_coeff(self, id_coeff:int):
        assert id_coeff<=self.deg and id_coeff>=0, ValueError(
            f"Expected a coefficient the polynom has, got {id_coeff}.")
        return self.coeffs[id_coeff]

    def set_deg(self, deg:int):
        assert deg>=0, ValueError(
            f"Expected a positive polynom degree, got {deg}.")
        if deg < self.deg:
            self.coeffs = self.coeffs[: deg]
        else:
            self.coeffs = np.append(self.coeffs, np.zeros(deg - self.deg))
        self.deg = deg
    
    def get_deg(self):
        return self.deg

    def derive(self):
        if self.deg == 0:
            self.coeffs = np.array([0.0])
        else:
            deriv = np.empty(self.deg)
            for i in range(1, self.deg+1):
                deriv[i - 1] = i * self.coeffs[i]

            self.deg -= 1
            self.coeffs = deriv
    
    def add(self, p: np.ndarray):
        n = max(self.deg, p.get_deg())
        res = np.zeros(n+1)
        for i in range(self.deg + 1):
            res[i] += self.coeffs[i]
        for i in range(p.get_deg() + 1):
            res[i] += p.coeffs[i]

        last_non_zero = 0
        for i in reversed(range(n + 1)):
            if abs(res[i]) > 1e-14:
                last_non_zero = i
                break
        
        self.deg = last_non_zero
        self.coeffs = res[:last_non_zero + 1]

    def mul_scalar(self, scalar: float):
        self.coeffs *= scalar
        if scalar == 0:
            self.deg = 0

    def mul_X(self):
        res = np.zeros(self.deg + 2)
        for i in range(self.deg + 1):
            res[i + 1] = self.coeffs[i]

        self.deg += 1
        self.coeffs = res

    def copy(self):
        p = Polynom(self.deg)
        p.coeffs = np.copy(self.coeffs)
        return p

class AssociatedLegendrePolynomsCalculator:
    """Class for associated Legendre Polynoms for m=2 (used for gravitationnal weak lensing).
    """
    def __init__(self, nb_l:int):
        self.P2 = Polynom(np.array([3, 0, -3]))
        self.P3 = Polynom(np.array([0, 15, 0, -15]))
        self.polynoms = [self.P2, self.P3]
        self.polynoms = self(nb_l)
    
    def __str__(self):
        return f"<AssociatedLegendrePolynoms n={len(self.polynoms)}>"
    
    def __call__(self, nb_l:int):
        assert nb_l>=0, ValueError(
            f"Expected l>=0, got {nb_l}.")

        if nb_l<=2:
            return self.polynoms[:nb_l]
        
        P0 = self.P2.copy()
        P1 = self.P3.copy()
        for l in range(3, nb_l+1):
            P_next = self.next_assoc_legendre(l, 2, P0, P1)
            P0, P1 = P1, P_next
            self.polynoms.append(P_next)
        return self.polynoms

    @staticmethod
    def next_assoc_legendre(l:int, m:int, P0:Polynom, P1:Polynom):
        P = Polynom(P1.get_deg()+1)

        a = P1.copy()
        b = P0.copy()

        a.mul_scalar((2*l + 1) / (l - m + 1))
        a.mul_X()

        b.mul_scalar(-(l + m) / (l - m + 1))

        P.add(a)
        P.add(b)
        return P
