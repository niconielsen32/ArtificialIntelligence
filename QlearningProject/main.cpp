#include <iostream>
#include "Qlearning.h"

using namespace std;

int main()
{
    Qlearning ql(30000);

    ql.train();

    ql.displayTrainedQTable();

    ql.deployAgent();

    return 0;
}