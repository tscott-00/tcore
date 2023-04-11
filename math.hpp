/**
 * @file   math.hpp
 * @author Thomas Scott
 * @brief  Provides universal constants, integration quadratures, and polynomial basis functions
 * 
 * @copyright Copyright (c) 2023
 */

#pragma once

#include <complex>

// Chopping block!
#include <unordered_map>
#include <math.h>
#include <utility>

#define PI  4.0*atan(1.0)
#define EPS pow(2.0, -52)

/**
 * @brief Provides weights and sample coordinates for quadratures used in numerical integration
 */
class Quadrature {
public:
    enum class Type {
        // TODO: Rename to GUASS_LEGENDRE
        GUASS,     // Accurate on polynomials up to degree [size*2-1]
        TRAPEZOID,
        COUNT
    } type;
    const int size;
    // Sample points (bounded in [x0, x1]) and sample weights interwoven
    const double *xw;

    /**
     * @brief Construct a new Quadrature object
     * 
     * @param size The number of samples present in the quadrature
     * @param x0   The lower bound of the sample space
     * @param x1   The upper bound of the sample space
     * @param type Which quadrature to use
     */
    Quadrature(Type type, int size, double x0, double x1);

    ~Quadrature();
    Quadrature(Quadrature&)                  = delete;
    Quadrature& operator=(const Quadrature&) = delete;

    /**
     * @brief Get the quadrature sample coordinate at a given index
     * 
     * @param i       The index in the quadrature
     * @return double The weight for the ith quadrature point
     */
    double x(int i) const {
        return xw[i*2];
    }

    /**
     * @brief Get the quadrature sample weight at a given index
     * 
     * @param i       The index in the quadrature
     * @return double The weight for the ith quadrature point
     */
    double w(int i) const {
        return xw[i*2+1];
    }
};

/**
 * @brief Provides evaluations for sets of polynomial basis functions with linear reference space,
 *        for both linearly spaced coordinates and on arbitrary quadratures
 */
class LinearRefBases {
public:
    typedef std::complex<double> (*EvalTBasis)(int, int, std::complex<double>);

    enum class Type {
        LAGRANGE,
        LEGENDRE,
        BERNSTEIN,
        COUNT
    };

    const Type          type;
    const int           basisCount;
    const double *const coeffs;
private:
    double     *evals;
    const bool  ownsEvals;
public:

    /**
     * @brief Construct a new linear reference bases object
     * 
     * @param type       The type of basis function to use
     * @param degree     The degree of the polynomial, leads to \p degree + 1 basis functions
     * @param qat        Defines the set of reference coordinates to be accessed later, if null
     *                       then \p degree + 1 number of linearly spaced coordinates will be used
     * @param coeffs     Defines the coefficients to be used to weigh the basis functions,
     *                       if null then VX(ref) can not be used, only BX(ref)
     * @param blockCache If false, the evaluation will be cached to speed up future constructions
     * @param h          The step sized to use for evaluating the derivatives with the complex step method
     */
    LinearRefBases(Type type, int degree, const Quadrature *qat = nullptr, double *coeffs = nullptr, bool blockCache = false, double h = 1.0E-200);
    
    /**
     * @brief Construct a new linear reference bases object
     * 
     * @param type   The type of basis function to use
     * @param degree The degree of the polynomial, leads to \p degree + 1 basis functions
     * @param qat    Defines the set of reference coordinates to be accessed later
     */
    LinearRefBases(Type type, int degree, const Quadrature &qat);

    ~LinearRefBases();
    LinearRefBases(LinearRefBases&)                  = delete;
    LinearRefBases& operator=(const LinearRefBases&) = delete;

    /**
     * @brief Evaluate a basis function at a reference point
     * 
     * @param basis   Which basis function to evaluate
     * @param ref     Where to evaluate the basis function, as defined by qat in initialization
     * @return double The value of the basis function
     */
    double B0(int basis, int ref) {
        return evals[basis*2 + ref*basisCount*2];
    }

    /**
     * @brief Evaluate the first derivative of a basis function at a reference point
     * 
     * @param basis   Which basis function to evaluate
     * @param ref     Where to evaluate the basis function, as defined by qat in initialization
     * @return double The first derivative of the basis function
     */
    double B1(int basis, int ref) {
        return evals[basis*2+1 + ref*basisCount*2];
    }

    /**
     * @brief Evaluate the value of the polynomial at a reference point
     * 
     * @param ref     Where to evaluate the 1D polynomial, as defined by qat in initialization
     * @return double The value of the 1D polynomial
     */
    double V0(int ref) {
        double V0 = 0.0;
        for (int basis = 0; basis < basisCount; ++basis)
            V0 += coeffs[basis]*evals[basis*2+1 + ref*basisCount*2];

        return V0;
    }

    /**
     * @brief Evaluate the first derivative of the 1D polynomial at a reference point
     * 
     * @param ref     Where to evaluate the polynomial, as defined by qat in initialization
     * @return double The first derivative of the 1D polynomial
     */
    double V1(int ref) {
        double V1 = 0.0;
        for (int basis = 0; basis < basisCount; ++basis)
            V1 += coeffs[basis]*evals[basis*2+1 + ref*basisCount*2];

        return V1;
    }

    /**
     * @brief Evaluate Lagrange basis polynomials for linear reference space
     * 
     * @param basis                 The basis polynomial index to evaluate, 0 <= \p basis < \p basisCount
     * @param basisCount            The number of basis functions in the set
     * @param point                 The point at which to evaluate, must be within [0.0, 1.0]
     * @return std::complex<double> The value of the basis polynomial
     */
    static std::complex<double> EvalLagrangeBasis(int basis, int basisCount, std::complex<double> point);

    /**
     * @brief Evaluate Legendre basis polynomials
     * 
     * @param basis                 The basis polynomial index to evaluate, value also corresponds to the degree
     * @param basisCount            The number of basis functions in the set (ignored)
     * @param point                 The point at which to evaluate, must be within [0.0, 1.0]
     * @return std::complex<double> The value of the basis polynomial
     */
    static std::complex<double> EvalLegendreBasis(int basis, int basisCount, std::complex<double> point);
    
    
    static std::complex<double> EvalBernsteinBasis(int basis, int basisCount, std::complex<double> point);
};

/**
 * @brief Provides evaluations for sets of polynomial basis functions on any reference space and coefficient set
 */
class WarpedRefBases {
public:
    enum class Type {
        LAGRANGE,
        B_SPLINE, // Makes a spline for 1D, Bernstein Bases/Beizer Curve for 1D if refs are linear, and NURBS for higher dimensions
        COUNT
    };

    const Type          type;
    const int           degree;
    const double *const evals;

    /**
     * @brief Construct a new warped reference bases object
     * 
     * @param type   The type of basis functions to use
     * @param degree The polynomial degree
     * @param qat    Defines the set of reference coordinates to be accessed later
     * @param refs   The reference points to use (knots for B-Spline, interpolation points for Lagrange),
     *                   null leads to linearly reference geometry on [0.0, 1.0], otherwise,
     *                   must contain \p degree - 1 values (preceeding and postceding assumed to be 0.0 and 1.0, respectively)
     * @param coeffs The coefficients to use on the basis functions (interpolants for Lagrange and control points for B-Spline),
     *                   null leads to linearly spaced values on [0.0, 1.0], otherwise,
     *                   must contain \p degree + 1 values for Lagrange or 2 * \p degree - 2 values for B-Spline
     * @param h      The step sized to use for evaluating the derivatives with the complex step method
     */
    WarpedRefBases(Type type, int degree, const Quadrature &qat, double *refs = nullptr, double *coeffs = nullptr, double h = 1.0E-200);

    ~WarpedRefBases();
    WarpedRefBases(WarpedRefBases&) = delete;
    WarpedRefBases& operator=(const WarpedRefBases&) = delete;

    /**
     * @brief Evaluate the value of the 1D polynomial at a reference point
     * 
     * @param ref     Where to evaluate the polynomial, as defined by qat in initialization
     * @return double The value of the 1D polynomial
     */
    double V0(int ref) {
        return evals[ref*2];
    }

    /**
     * @brief Evaluate the first derivative of the 1D polynomial at a reference point
     * 
     * @param ref     Where to evaluate the polynomial, as defined by qat in initialization
     * @return double The first derivative of the 1D basis function
     */
    double V1(int ref) {
        return evals[ref*2+1];
    }
};
