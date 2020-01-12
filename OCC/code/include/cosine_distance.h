#ifndef COSINE_DISTANCE_H
#define COSINE_DISTANCE_H

#include <mlpack/core/kernels/cosine_distance.hpp>

class MyCosineDistance
{
    public:

        MyCosineDistance(){};

        template<typename VecTypeA, typename VecTypeB>
        static double Evaluate(const VecTypeA& a, const VecTypeB& b)
        {
            return 1 - mlpack::kernel::CosineDistance::Evaluate(a, b); // Despite the name, mlpack computes cosine similarity.
        }

        template<typename Archive>
        void serialize(Archive& /* ar */, const unsigned int /* version */) { }
};

#endif