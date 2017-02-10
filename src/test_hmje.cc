#include "catch.hpp"
#include "tvb/hmje.h"


TEST_CASE("hmje model", "[hmje]")
{
    float r_value = 0.0025;
    float r_default = 0.00035f;

    using model_type = tvb::hmje<4, float>;
    model_type model;

    REQUIRE(model.r() == r_default);

    model.r() = r_value;

    REQUIRE(model.r() == r_value);
    REQUIRE(model_type::state_type::width() == 4);
    REQUIRE(model_type::state_type::length() == 6);
    REQUIRE(model_type::coupling_type::width() == 4);
    REQUIRE(model_type::coupling_type::length() == 2);

    REQUIRE(sizeof(model_type) == (model_type::n_param() * sizeof(model_type::value_type)));

    model_type::state_type state, deriv;
    model_type::coupling_type coupling;
    model.eval(state, deriv, coupling);

    // TODO run with different x0 ensure seizure
}
