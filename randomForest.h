#ifndef _RANDOMFOREST_H_
#define _RANDOMFOREST_H_
#include <vector>
#include "decisionTree.h"

std::vector<std::vector<int>> get_random_samples(
    const std::vector<std::vector<int>> &samples, int num_to_return);

class RandomForest {
 protected:
    std::vector<Node> trees;
    int num_trees;
    std::vector<std::vector<int>> images;

 public:
    RandomForest(int, const std::vector<std::vector<int>> &);
    void build();
    int predict(const std::vector<int> &);
};

#endif
