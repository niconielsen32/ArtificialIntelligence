#include <iostream>
#include <qlearning.h>

using namespace std;

int main()
{
    Qlearning ql(30000);

    ql.train();

    ql.displayTrainedQTable();

    ql.deployAgent2();

    return 0;
}
