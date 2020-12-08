#include "qlearning.h"



using namespace std;

template<typename Iter, typename RandomGenerator>
Iter select_randomly_rd(Iter start, Iter end, RandomGenerator& g) {
    uniform_int_distribution<> dis(0, distance(start, end) - 1);
    advance(start, dis(g));
    return start;
}

template<typename Iter>
Iter select_randomly(Iter start, Iter end) {
    static random_device rd;
    static mt19937 gen(rd());
    return select_randomly_rd(start, end, gen);
}


int findKthLargestIndex(vector<double> actions, int kth){

    double kthNumber;
    vector<double> tempVec = actions;
    sort(tempVec.begin(), tempVec.end(), greater<double>());

    kthNumber = tempVec[kth];

    for(size_t i = 0; i < actions.size(); i++){
        if(actions[i] == double(kthNumber)){
            return i;
        }
    }

    return -1;
}

Qlearning::Qlearning()
{

}


// Initialize the q-table and the environment
void Qlearning::initQTable(){

    vector<vector<double> > q_table_init(numberOfStates,std::vector<double>(numberOfStates, 0.0));

    q_table = q_table_init;

}


void Qlearning::initRewardMatrix(){

    // State init with distributed marbles in rooms - 1 marble is 5 points
    Room room0Reward(25);
    rooms.push_back((room0Reward));
    Room room1Reward(5);
    rooms.push_back(room1Reward);
    Room room2Reward(5);
    rooms.push_back(room2Reward);
    Room room3Reward(15);
    rooms.push_back(room3Reward);
    Room room4Reward(10);
    rooms.push_back(room4Reward);
    Room room5Reward(10);
    rooms.push_back(room5Reward);
    Room room6Reward(10);
    rooms.push_back(room6Reward);
    Room room7Reward(20);
    rooms.push_back(room7Reward);
    Room room8Reward(5);
    rooms.push_back(room8Reward);
    Room room9Reward(10);
    rooms.push_back(room9Reward);
    Room room10Reward(25);
    rooms.push_back(room10Reward);


    // Reward matrix
    vector<int> room0 = {-1, room1Reward.reward, -1, -1, -1, -1, -1, -1, -1, -1, -1};
    vector<int> room1 = {room0Reward.reward, -1, room2Reward.reward, -1, -1, room5Reward.reward, -1, -1, -1, -1, -1};
    vector<int> room2 = {-1, room1Reward.reward, -1, room3Reward.reward, room4Reward.reward, room5Reward.reward, -1, -1, -1, -1, -1};
    vector<int> room3 = {-1, -1, room2Reward.reward, -1, -1, -1, -1, -1, -1, -1, -1};
    vector<int> room4 = {-1, -1, room2Reward.reward, -1, -1, -1, -1, -1, -1, -1, -1};
    vector<int> room5 = {-1, room1Reward.reward, room2Reward.reward, -1, -1, -1, room6Reward.reward, room7Reward.reward, room8Reward.reward, room9Reward.reward, -1};
    vector<int> room6 = {-1, -1, -1, -1, -1, room5Reward.reward, -1, -1, -1, -1, -1};
    vector<int> room7 = {-1, -1, -1, -1, -1, room5Reward.reward, -1, -1, -1, -1, -1};
    vector<int> room8 = {-1, -1, -1, -1, -1, room5Reward.reward, -1, -1, -1, -1, -1};
    vector<int> room9 = {-1, -1, -1, -1, -1, room5Reward.reward, -1, -1, -1, -1, room10Reward.reward};
    vector<int> room10 = {-1, -1, -1, -1, -1, -1, -1, -1, -1, room9Reward.reward, -1};


    reward_matrix.push_back(room0);
    reward_matrix.push_back(room1);
    reward_matrix.push_back(room2);
    reward_matrix.push_back(room3);
    reward_matrix.push_back(room4);
    reward_matrix.push_back(room5);
    reward_matrix.push_back(room6);
    reward_matrix.push_back(room7);
    reward_matrix.push_back(room8);
    reward_matrix.push_back(room9);
    reward_matrix.push_back(room10);

     for(int state = 0; state < numberOfStates; state++){
         for(int action = 0; action < numberOfStates; action++){
             if(reward_matrix[state][action] != -1){
                 rooms[state].numberOfActions++;
             }
         }
     }

}



// Get random init state
void Qlearning::setRandomInitState(bool random, int state){

    random_device rd;
    uniform_int_distribution<int> randomState(0, numberOfStates-1);

    // Markov property - keeping track of which rooms have been visited
    for(int i = 0; i < numberOfStates; i++){
        rooms[i].roomNumber = i;
        rooms[i].isVisited = false;
        rooms[i].numberOfVisits = 0;
    }

    if(random){
        initial_state_random = randomState(rd);
    } else {
        initial_state_random = state;
    }


    rooms[initial_state_random].isVisited = true;

}



// Get an action from the given state
int Qlearning::getAction(int current_state){

    vector<int> valid_actions = {};
    for(size_t action = 0; action < reward_matrix[current_state].size(); action++){
        if(reward_matrix[current_state][action] != -1){
            valid_actions.push_back(action);
        }
    }

    std::random_device rd;
    uniform_int_distribution<int> dist(0, valid_actions.size() - 1);
    uniform_real_distribution<double> exploration(0.0, 1.0);

    double exploration_rate_threshold = exploration(rd);


    if(exploration_rate_threshold > epsilon){

        if(rooms[current_state].numberOfVisits == 0){
            cout << "exploiting1" << endl;
            rooms[current_state].numberOfVisits += 1;
            return std::distance(q_table[current_state].begin(), max_element(q_table[current_state].begin(), q_table[current_state].end()));
        } else if(rooms[current_state].numberOfVisits <= rooms[current_state].numberOfActions) {


            if(rooms[current_state].numberOfVisits == rooms[current_state].numberOfActions){

                rooms[current_state].numberOfVisits = 0;
                cout << "exploiting3: " << endl;
            }

            int kth = rooms[current_state].numberOfVisits;

            cout << "kth: " << kth << endl;
            int indexLargest = findKthLargestIndex(q_table[current_state], kth);
            rooms[current_state].numberOfVisits += 1;
            cout << "exploiting2: " << endl;



            return indexLargest;

        }
    }
    return valid_actions[dist(rd)];
}



// Get and take an action
int Qlearning::takeAction(int current_state, bool display){

    // Take a new action from the current state
    int action = getAction(current_state);

    // Next state will be the action taken
    int new_state = action;

    // Get the current state action value
    double current_sa_value = q_table[current_state][action];
    // Get the reward for the given state and the action taken
    double reward = rooms[action].reward;

    // If the room has already been visited - reward is set to 0
    if(rooms[action].isVisited == true){
        reward = 0;
    }

    cout << "Reward: " << reward << endl;
    // Sum the rewards for the actions taken
    maxRewardRecieved += reward;

    // Max future reward
    auto max = max_element(q_table[new_state].begin(), q_table[new_state].end());
    double future_sa_reward = *max;

    // Q learning equation
    double Q_value_current_state = current_sa_value + learning_rate * (reward + gamma * future_sa_reward - current_sa_value);

    // Update Q-table with new q value
    q_table[current_state][action] = Q_value_current_state;

    if(display){
        cout << "Old state: " << current_state << " | " << "New state: " << new_state << endl;
    }

    // Mark room as visited
    rooms[current_state].isVisited = true;

    cout << endl;

    // Return the new state from taken action
    return new_state;
}



// Run one episode
void Qlearning::runEpisode(){

    int current_state = initial_state_random;
    int stepCount = 0;

    int initreward = rooms[current_state].reward;
    cout << "init reward: " << initreward << endl;
    maxRewardRecieved += initreward;

    while(true){
         // Take action from given state
         current_state = takeAction(current_state, true);
         stepCount++;
         // Terminate after max number of steps
         if(stepCount == maxStepsPerEpisode){
             cout << "Episode ended!" << endl;
             cout << "Max Reward: " << maxRewardRecieved << endl;

             expectedReturnPerEpisode.push_back(maxRewardRecieved);
             maxRewardRecieved = 0;
             break;
         }
    }
}



// Train the agent - Run several episodes
void Qlearning::train(){

    // Init the Q table and reward matrix
    initQTable();
    initRewardMatrix();

    cout << "Training started..." << endl;
    for (int episode = 0; episode < episodes; episode++) {
        episodeVec.push_back(episode);
        cout << "Episode: " << episode << endl;
        //int current_state = 8;
        // Set random state - get the reward and mark as visitied
        setRandomInitState(true, 0);

        // Run an episode
        runEpisode();

        // vector with epsilon for tests
        epsilonVec.push_back(epsilon);
        learningRateVec.push_back(learning_rate);

        // Update the epsilon value so it starts out exploring a lot and then less and less
        epsilon = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * exp(-epsilon_decay * episode);
        //learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * exp(-learning_rate_decay * episode);
        //cout << "epsilon: " << epsilon << endl;
    }

    cout << "Training done." << endl;

    // Output data for episodes and rewards to csv file
    dataToCSV();

}



void Qlearning::displayTrainedQTable(){

    // Show the trained Q table
    for(int state = 0; state < numberOfStates; state++){
        for(int action = 0; action < numberOfStates; action++){
            std::cout << q_table[state][action] << " ";
        }
       std::cout << std::endl;
    }
}


void Qlearning::deployAgent2(){

    cout << "Deploy agent" << endl;
    // Epsilon as 0 so it only exploit what it has learned during training
    epsilon = 0;
    // Set init stata - get reward and mark as visited
    setRandomInitState(true, 0);
    // Run 1 episode
    runEpisode();

}



void Qlearning::deployAgent(){

    q_tableDeploy = q_table;

    cout << "Deploy agent" << endl;

    initRewardMatrix();
    //setRandomInitState();

    int current_state = initial_state_random;
    int stepCount = 0;
    int action;

    cout << "max: " << maxRewardRecieved << endl;
    double reward = 0;


    while(true){
         action = std::distance(q_tableDeploy[current_state].begin(), max_element(q_tableDeploy[current_state].begin(), q_tableDeploy[current_state].end()));
         cout << "Old state: " << current_state << " | " << "New state: " << action << endl;
         q_tableDeploy[current_state][action] = 0;

         // Markov property - keeping track of which rooms have been visited
         for(int i = 0; i < numberOfStates; i++){
             rooms[i].roomNumber = i;
             rooms[i].isVisited = false;
             rooms[i].numberOfVisits = 0;
         }


         reward = reward_matrix[current_state][action];

         if(rooms[current_state].isVisited == true){
             reward = 0;
         }

         cout << reward << endl;
         maxRewardRecieved += reward;

         rooms[current_state].isVisited = true;

         current_state = action;

         stepCount++;
         if(stepCount == maxStepsPerEpisode){
             cout << "Episode ended!" << endl;
             cout << "Reward: " << maxRewardRecieved << endl;
             expectedReturnPerEpisode.push_back(maxRewardRecieved);
             maxRewardRecieved = 0;

             break;
         }

    }

}



void Qlearning::dataToCSV(){

    outputFile.open(filename);

    for(auto &episode : episodeVec){
        outputFile << episode << ",";
    }

    outputFile << endl;

    for(auto &returnEpisode : expectedReturnPerEpisode){
        outputFile << returnEpisode << ",";
    }

    outputFile << endl;

    for(auto &epsilon : epsilonVec){
        outputFile << epsilon << ",";
    }

    outputFile << endl;

    for(auto &learning_rate : learningRateVec){
        outputFile << learning_rate << ",";
    }

    outputFile.close();


    cout << "Data formatted to CSV file." << endl;


}
