#include <tuple>
#include "benchmark/benchmark.h"
#include "tvb/rww.h"
#include "tvb/hmje.h"
#include "tvb/linde.h"
#include "tvb/euler.h"

template <typename model_type>
static void model_step(benchmark::State& bm_state)
{
    using scheme_type = tvb::euler<model_type>;
    model_type model;
    scheme_type scheme;
    typename model_type::state_type state;
    typename model_type::coupling_type coupling;
    while (bm_state.KeepRunning())
    {
        scheme.eval(state, coupling, model);
    }
}

BENCHMARK_TEMPLATE(model_step, tvb::rww<1, float> );
BENCHMARK_TEMPLATE(model_step, tvb::rww<2, float>);
BENCHMARK_TEMPLATE(model_step, tvb::rww<4, float>);
BENCHMARK_TEMPLATE(model_step, tvb::rww<8, float>);
BENCHMARK_TEMPLATE(model_step, tvb::rww<16, float>);
BENCHMARK_TEMPLATE(model_step, tvb::rww<32, float>);
BENCHMARK_TEMPLATE(model_step, tvb::rww<64, float>);

BENCHMARK_TEMPLATE(model_step, tvb::hmje<1, float> );
BENCHMARK_TEMPLATE(model_step, tvb::hmje<2, float>);
BENCHMARK_TEMPLATE(model_step, tvb::hmje<4, float>);
BENCHMARK_TEMPLATE(model_step, tvb::hmje<8, float>);
BENCHMARK_TEMPLATE(model_step, tvb::hmje<16, float>);
BENCHMARK_TEMPLATE(model_step, tvb::hmje<32, float>);
BENCHMARK_TEMPLATE(model_step, tvb::hmje<64, float>);

BENCHMARK_TEMPLATE(model_step, tvb::linde<1, float> );
BENCHMARK_TEMPLATE(model_step, tvb::linde<2, float>);
BENCHMARK_TEMPLATE(model_step, tvb::linde<4, float>);
BENCHMARK_TEMPLATE(model_step, tvb::linde<8, float>);
BENCHMARK_TEMPLATE(model_step, tvb::linde<16, float>);
BENCHMARK_TEMPLATE(model_step, tvb::linde<32, float>);
BENCHMARK_TEMPLATE(model_step, tvb::linde<64, float>);

BENCHMARK_MAIN();
