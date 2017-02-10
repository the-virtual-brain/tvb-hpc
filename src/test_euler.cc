#include "catch.hpp"
#include "tvb/rww.h"
#include "tvb/linde.h"
#include "tvb/euler.h"


TEST_CASE("euler model", "[euler]")
{
    using model_type = tvb::linde<4, float>;
    using scheme_type = tvb::euler<model_type>;

    model_type model;
    scheme_type scheme;

    model_type::state_type state;
    model_type::coupling_type coupling;

    coupling.fill(0);

    for (size_t i=0; i<state.width(); i++)
        state(0, i) = ((float) i);

    scheme.dt() = 0.2;
    scheme.eval(state, coupling, model);

    for (size_t i=0; i<state.width(); i++)
    {
        float init_cond = ((float) i);
        float final_cond = init_cond + model.lambda() * init_cond * scheme.dt();
        REQUIRE(state(0, i) == final_cond);
    }

    // TODO do sim in TVB w/ hmje, check latency of seizure onset with i.c.
}
