#include "catch.hpp"
#include "tvb/util.h"

TEST_CASE("utility functions", "[util]")
{
    SECTION("wrap behaves like Python modulo not C modulo") {
        // python -c 'print [(i, i%7) for i in range(-10, 10)]'
        int results[20][2] = {{-10, 4}, {-9, 5}, {-8, 6}, {-7, 0}, {-6, 1}, {-5, 2}, {-4, 3}, {-3, 4}, {-2, 5}, {-1, 6}, {0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 5}, {6, 6}, {7, 0}, {8, 1}, {9, 2}};
        for (int i=0; i<20; i++)
            REQUIRE(tvb::wrap<>(results[i][0], 7) == results[i][1]);
    }

    SECTION("chunk api") {

        tvb::chunk<4, 4, float> chunk;

        for (size_t i=0; i<chunk.length(); i++)
            for (size_t j=0; j<chunk.width(); j++)
                chunk(i, j) = ((float) i * 4 + j);

        for (size_t i=0; i<chunk.length(); i++)
            for (size_t j=0; j<chunk.width(); j++)
                REQUIRE(chunk(i, j) == ((float) i * 4 + j));

    }
}
