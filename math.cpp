#include "math.hpp"

#include <iostream>
#include <unordered_map>

std::complex<double> EvalLegendreBasis(int basis, int basisCount, std::complex<double> zeta) {
    // Compute using the recurrance relation
    int n = basis;
    std::complex<double> history[2] = { 1.0, zeta };
    for (int i = 0; i < basis-1; ++i) {
        std::complex<double> eval = ((2.0*i+3.0)*zeta*history[1] - (i+1.0)*history[0]) / (i+2.0);
        history[0] = history[1];
        history[1] = eval;
    }
    
    return history[std::min(basis, 1)];
}

double SolveLegendreRoot(int n, double x, double h, double limiter) {
    double PLast = 1;
    double P     = 0;
    for (int i = 0; i < 100; ++i) {
        if (abs(P-PLast) < EPS)
            break;
        PLast = P;
        P             = EvalLegendreBasis(n, n, std::complex<double>(x  )).real();
        double dP__dx = EvalLegendreBasis(n, n, std::complex<double>(x,h)).imag()/h;
        x -= std::min(std::max(P/dP__dx, -limiter), limiter);
    }

    return x;
}

double* GuassQuadratureGenerator(int N, double r0, double r1) {
    // Tested against lgwt.m for N <= 75
    if (N > 75)
        std::cerr << "WARNING: Guass quadrature weights and coordinates are likely to be outside of machine precision for N > 75!" << std::endl;

    double h = 1.0E-200; // TODO: Take as a parameter?
    double* xw = new double[N*2];
    // Compute Legendre roots x_i by using near-equally spaced arccos(x_i) as initial guesses (common strategy)
    double x_g = 1.0;
    double limiter = 0.1 / N;
    for (int i = 0; i < N; ++i) {
        double x_g0 = x_g;
        xw[i*2] = SolveLegendreRoot(N, x_g, h, limiter);

        // Keep looking if it found the same root twice
        if (i > 0)
            while (abs(xw[i*2] - xw[(i-1)*2]) <= EPS) {
                x_g -= 0.1*PI / static_cast<double>(N-1);
                xw[i*2] = SolveLegendreRoot(N, x_g, h, limiter);
            }
        // Guess the next x location
        x_g = cos(2.0*acos(xw[i*2]) - acos(x_g0));
    }

    // Compute weights
    for (int i = 0; i < N; ++i) {
        double dP__dx = EvalLegendreBasis(N, N, std::complex<double>(xw[i*2],h)).imag()/h;
        xw[i*2+1] = 2.0 / ((1.0 - xw[i*2]*xw[i*2]) * dP__dx*dP__dx);
    }

    // Move from [-1, 1] to [r0, r1]
    for (int i = 0; i < N; ++i) {
        xw[i*2  ] = xw[i*2  ] * (r1-r0)/2.0 + (r1+r0)/2.0;
        xw[i*2+1] = xw[i*2+1] * (r1-r0)/2.0;
    }

    return xw;
}

double* TrapezoidQuadratureGenerator(int N, double r0, double r1) {
    double* xw = new double[N*2];
    for (int i = 0; i < N; ++i) {
        xw[i*2] = static_cast<double>(i)/(N-1);
        xw[i*2+1] = (i==0||i==N-1)?0.5:1.0;
    }

    return xw;
}

Quadrature::Quadrature(Type type, int size, double x0, double x1) :
    type(type),
    size(size),
    xw(nullptr) {
    
    switch (type) {
    case Type::GUASS:
        xw = GuassQuadratureGenerator(size, x0, x1);
        break;
    case Type::TRAPEZOID:
        xw = TrapezoidQuadratureGenerator(size, x0, x1);
        break;
    default:
        // TODO: Assert and use custom error message handler!
        std::cerr << "Unknown quadrature!" << std::endl;
        break;
    }
}

Quadrature::~Quadrature() {
    if (xw == nullptr) return;
    delete[] xw;
}

// Internal cache system w/ memory management
class LinearRefBasesCache {
private:
    std::unordered_map<const Quadrature *, std::unordered_map<int, double *>> entries[static_cast<int>(LinearRefBases::Type::COUNT)];
public:
    LinearRefBasesCache() { }

    double * GetEntry(LinearRefBases::Type type, const Quadrature *qat, int basisCount, bool& mustPopulate) {
        auto& qatPair = entries[static_cast<int>(type)][qat];
        auto entry = qatPair.find(basisCount);
        if (entry == qatPair.end()) {
            qatPair[basisCount] = new double[qat != nullptr ? 2*basisCount*qat->size : 2*basisCount*basisCount];
            mustPopulate = true;

            return qatPair[basisCount];
        }
        
        mustPopulate = false;
        
        return entry->second;
    }

    ~LinearRefBasesCache() {
        for (int type = 0; type < static_cast<int>(LinearRefBases::Type::COUNT); ++type)
            for (auto& qatPair : entries[type])
                for (auto& numPair : qatPair.second)
                    delete[] numPair.second;
    }
};

LinearRefBasesCache linRefBasesCache;

LinearRefBases::LinearRefBases(Type type, int degree, const Quadrature *qat, double *coeffs, bool blockCache, double h) :
    type(type),
    basisCount(degree+1),
    coeffs(coeffs),
    evals(nullptr),
    ownsEvals(blockCache) {

    bool mustPopulate = true;
    if (blockCache)
        evals = new double[qat != nullptr ? 2*basisCount*qat->size : 2*basisCount*basisCount];
    else
        evals = linRefBasesCache.GetEntry(type, qat, basisCount, mustPopulate);

    EvalTBasis basisFunc;
    switch (type) {
    case Type::LAGRANGE:
        basisFunc = &EvalLagrangeBasis;
        break;
    case Type::LEGENDRE:
        basisFunc = &EvalLegendreBasis;
        break;
    case Type::BERNSTEIN:
        basisFunc = &EvalBernsteinBasis;
        break;
    default:
        basisFunc = &EvalLagrangeBasis;
        // TODO: Assert and error log!
        std::cerr << "Invalid basis type, using Lagrange instead..." << std::endl;
        break;
    }

    if (mustPopulate) {
        if (qat == nullptr)
            for (int eval = 0; eval < basisCount; ++eval) {
                double ref = static_cast<double>(eval)/(basisCount-1);
                for (int basis = 0; basis < basisCount; ++basis) {
                    evals[basis*2   + eval*basisCount*2] = basisFunc(basis, basisCount, std::complex<double>(ref  )).real();
                    evals[basis*2+1 + eval*basisCount*2] = basisFunc(basis, basisCount, std::complex<double>(ref,h)).imag()/h;
                }
            }
        else
            for (int eval = 0; eval < qat->size; ++eval) {
                // TODO: Properly map from qat range to basis range!
                double ref = qat->x(eval);
                for (int basis = 0; basis < basisCount; ++basis) {
                    evals[basis*2   + eval*basisCount*2] = basisFunc(basis, basisCount, std::complex<double>(ref  )).real();
                    evals[basis*2+1 + eval*basisCount*2] = basisFunc(basis, basisCount, std::complex<double>(ref,h)).imag()/h;
                }
            }
    }
}

LinearRefBases::LinearRefBases(Type type, int degree, const Quadrature &qat) :
    type(type),
    basisCount(degree+1),
    coeffs(nullptr),
    evals(nullptr),
    ownsEvals(false) {
    
    new (this) LinearRefBases(type, degree, &qat);
}

LinearRefBases::~LinearRefBases() {
    if (ownsEvals)
        delete[] evals;
}

std::complex<double> LinearRefBases::EvalLagrangeBasis(int basis, int basisCount, std::complex<double> point) {
    std::complex<double> phi(1.0);
    // Doing this is equivalent to dividing others by (valueCount-1) to normalize
    for (int i = 0; i < basisCount; ++i)
        if (i != basis)
            phi *= (point*static_cast<double>(basisCount-1)-static_cast<double>(i))/static_cast<double>(basis-i);
    
    return phi;
}

std::complex<double> LinearRefBases::EvalLegendreBasis(int basis, int basisCount, std::complex<double> point) {
    // Map from [0.0, 1.0] to [-1.0, 1.0]
    point = point * 2.0 - 1.0;
    // Compute using the recurrance relation
    int n = basis;
    std::complex<double> history[2] = { 1.0, point };
    for (int i = 0; i < basis-1; ++i) {
        std::complex<double> eval = ((2.0*i+3.0)*point*history[1] - (i+1.0)*history[0]) / (i+2.0);
        history[0] = history[1];
        history[1] = eval;
    }
    
    return history[std::min(basis, 1)];
}

int Factorial(int i) {
    if (i == 0)
        return 1;
    else if (i > 1)
        return i * Factorial(i-1);
    else
        return i;
}

std::complex<double> LinearRefBases::EvalBernsteinBasis(int basis, int basisCount, std::complex<double> point) {
    int v = basis;
    int n = basisCount-1;
    double binomialCoeff = static_cast<double>(Factorial(n))/(Factorial(v)*Factorial(n-v));
    // Define 0^0 as 1.0
    std::complex<double> term0 = v > 0 ? pow(point,static_cast<double>(v)) : 1.0;
    std::complex<double> term1 = (n-v) > 0 ? pow(1.0-point,static_cast<double>(n-v)) : 1.0;

    return binomialCoeff*term0*term1;
}
