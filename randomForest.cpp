#include "randomForest.h"
#include <iostream>
#include <random>
#include <vector>
#include <string>
#include "decisionTree.h"

using std::vector;
using std::pair;
using std::string;
using std::mt19937;

/*
 * Intoarce un vector de marime num_to_return cu elemente random,
 * diferite din samples
 */
vector<vector<int>> get_random_samples(const vector<vector<int>> &samples,
                                       int num_to_return) {
                                           
    vector<vector<int>> ret;
    int size = samples.size();
    vector<int> frequency(size, 0);
    std::uniform_int_distribution<int> d(0, num_to_return - 1);
    std::random_device rd;

    /*
     * Se genereaza o valoare.
     *  Daca valoarea este diferita de 0
     * si nu a mai fost generata anterior,
     * se retine aparitia acesteia.
     * In caz contrar,
     * generarea nu este luata in considerare
     */
    for (int i = 0; i < num_to_return; ++i) {
        int value = d(rd);

        if (!frequency[value]) {
            frequency[value] = 1;
            ret.push_back(samples[value]);
        } else {
            --i;
        }
    }

    return ret;
}

RandomForest::RandomForest(int num_trees, const vector<vector<int>> &samples)
    : num_trees(num_trees), images(samples) {}

/*
 * Aloca pentru fiecare Tree cate n / num_trees
 * Unde n e numarul total de teste de training
 * Apoi antreneaza fiecare tree cu testele alese
 */
void RandomForest::build() {
    assert(!images.empty());
    vector<vector<int>> random_samples;

    int data_size = images.size() / num_trees;

    for (int i = 0; i < num_trees; i++) {
        random_samples = get_random_samples(images, data_size);

        /*
         * Construieste un Tree nou si il antreneaza
         */
        trees.push_back(Node());
        trees[trees.size() - 1].train(random_samples);
    }
}

/*
 * Va intoarce cea mai probabila prezicere pentru testul din argument
 * se va interoga fiecare Tree si se va considera raspunsul final ca
 * fiind cel majoritar
 */
int RandomForest::predict(const vector<int> &image) {
    vector<int> frequency(10, 0);

    /*
     * Se calculeaza numarul de aparitii al fiecarei cifre prezise
     */
    for (int i = 0; i < num_trees; ++i) {
        ++frequency[trees[i].predict(image)];
    }

    /*
     * Se cauta cifra care a fost prezisa de cele mai multe ori
     */
    int max = -1;
    int rez = -1;
    for (int i = 0; i < 10; ++i) {
        if (frequency[i] > max) {
            max = frequency[i];
            rez = i;
        }
    }

    return rez;
}
