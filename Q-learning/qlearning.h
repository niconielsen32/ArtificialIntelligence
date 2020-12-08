#ifndef QLEARNING_H
#define QLEARNING_H

#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <fstream>
#include <math.h>

using namespace std;


struct Room{

    int roomNumber;
    bool isVisited = false;
    int reward = 0;
    int numberOfVisits = 0;
    int numberOfActions = 0;

    Room();
    Room(int reward)
        : reward(reward){}

};


class Qlearning
{
public:
    Qlearning();
    Qlearning(int episodes)
        : episodes(episodes) {}

    // Initialize the q-table and the environment
    void initQTable();
    void initRewardMatrix();

    // Get random init state
    void setRandomInitState(bool random, int state);

    // Get and take an action
    int takeAction(int current_state, bool display);
    int getAction(int current_state);

    // Run one episode
    void runEpisode();

    // Train the agent - Run several episodes
    void train();

    void displayTrainedQTable();

    // Deploy the trained agent
    void deployAgent();
    void deployAgent2();

    void normalizeQTable();

    void dataToCSV();


private:

    vector<vector<double> > q_table;
    vector<vector<double> > q_tableDeploy;
    vector<vector<int> > reward_matrix;
    vector<Room> rooms;

    double maxRewardRecieved = 0;

    int numberOfStates = 11;
    int initial_state;
    int initial_state_random;

    float gamma = 0.3;
    float learning_rate = 0.01;

    float epsilon = 1.0;
    float epsilon_decay = 0.0001;


    float min_exploration_rate = 0.01;
    float max_exploration_rate = 1.0;

    int episodes;
    int maxStepsPerEpisode = 6;


    // Vectors for tests
    vector<int> episodeVec;
    vector<int> expectedReturnPerEpisode;
    vector<double> epsilonVec;
    vector<double> learningRateVec;

    ofstream outputFile;
    //string filename = "qlearningTestNew.csv";
    //string filename = "qlearningTestNewNo2.csv";
    string filename = "qlearningTestNewNo3.csv";

};

#endif // QLEARNING_H
