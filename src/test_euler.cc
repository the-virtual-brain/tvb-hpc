#include "catch.hpp"
#include "tvb/rww.h"
#include "tvb/euler.h"


TEST_CASE("euler model", "[euler]")
{
    float a_value = -3.42f;
    float a_default = 0.270f;

    using model_type = tvb::rww<4, float>;
    model_type model;

    using scheme_type = tvb::euler<model_type>;
    scheme_type scheme;

    scheme.dt() = 0.01;

    model_type::state_type state;
    model_type::coupling_type coupling;


}
