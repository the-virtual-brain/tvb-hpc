#include "catch.hpp"
#include "tvb/rww.h"


TEST_CASE("rww model", "[rww]")
{
    float a_value = -3.42f;
    float a_default = 0.270f;

    using model_type = tvb::rww<4, float>;
    model_type model;

    REQUIRE(model.a() == a_default);

    model.a() = a_value;

    REQUIRE(model.a() == a_value);
    REQUIRE(model_type::state_type::width() == 4);
    REQUIRE(model_type::state_type::length() == 1);
}
