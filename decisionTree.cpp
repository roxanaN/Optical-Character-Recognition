#include "./decisionTree.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using std::string;
using std::pair;
using std::vector;
using std::unordered_map;
using std::make_shared;

/*
 * structura unui nod din decision tree
 * splitIndex = dimensiunea in functie de care se imparte
 * split_value = valoarea in functie de care se imparte
 * is_leaf si result sunt pentru cazul in care avem un nod frunza
 */
Node::Node() {
	is_leaf = false;
	left = nullptr;
	right = nullptr;
}

void Node::make_decision_node(const int index, const int val) {
	split_index = index;
	split_value = val;
}

/*
 * Seteaza nodul ca fiind de tip frunza
 * is_single_class = true -> toate testele au aceeasi clasa
 * is_single_class = false -> se alege clasa care apare cel mai des
 */
void Node::make_leaf(const vector<vector<int>> &samples,
					 const bool is_single_class) {
	is_leaf = true;
	if (is_single_class) {
		result = samples[0][0];
	} else {
		/*
		 * Se calculeaza numarul de aparitii al fiecarei clase\
		 */
		vector<int> frequency(10, 0);
		int size = samples.size();
		for (int i = 0; i < size; ++i) {
			++frequency[samples[i][0]];
		}

		/*
		 * Se cauta clasa care apare de cele mai multe ori
		 */
		int max = frequency[0];
		int index = 0;
		for (int i = 1; i < 10; ++i) {
			if (frequency[i] > max) {
				max = frequency[i];
				index = i;
			}
		}

		result = index;
	}
}

/*
 *Intoarce cea mai buna dimensiune si valoare de split dintre testele
 * primite. Prin cel mai bun split (dimensiune si valoare)
 * ne referim la split-ul care maximizeaza IG
 * pair-ul intors este format din (split_index, split_value)
 */
pair<int, int> find_best_split(const vector<vector<int>> &samples,
							   const vector<int> &dimensions) {
	int splitIndex = -1, splitValue = -1;
	int size_d = dimensions.size();
	float max_IG = 0;
	float H_parent = get_entropy(samples);
	vector<int> uniqueValues;
	pair<vector<int>, vector<int>> subtrees;

	/*
	 * Se parcurge vectorul dimensions
	 */
	for (int i = 0; i < size_d; ++i) {
		/*
		 * Se determina valorile unice de pe coloana dimension[i]
		 */
		uniqueValues = compute_unique(samples, dimensions[i]);
		int dim = uniqueValues.size();

		/* 
		 * Se calculeaza Information Gain, pentru fiecare valoare unica
		 */
		for (int j = 0; j < dim; ++j) {
			subtrees = get_split_as_indexes(samples, dimensions[i],
											uniqueValues[j]);

			if (!subtrees.first.size() || !subtrees.second.size()) {
				continue;
			}

			float left_entropy = get_entropy_by_indexes(samples,
														subtrees.first);
			float right_entropy = get_entropy_by_indexes(samples,
														 subtrees.second);

			int left_size = subtrees.first.size();
			int right_size = subtrees.second.size();
			int n = left_size + right_size;

			float sum = left_size * left_entropy  + right_size * right_entropy;
			float IG = H_parent - sum / n;

			if (IG > max_IG) {
				max_IG = IG;
				splitIndex = dimensions[i];
				splitValue = uniqueValues[j];
			}
		}
	}

	return pair<int, int>(splitIndex, splitValue);
}

/*
 * Antreneaza nodul curent si copii sai, daca e nevoie
 * 1) verifica daca toate testele primite au aceeasi clasa (raspuns)
 * Daca da, acest nod devine frunza, altfel continua algoritmul.
 * 2) Daca nu exista niciun split valid, acest nod devine frunza. Altfel,
 * ia cel mai bun split si continua recursiv
 */
void Node::train(const vector<vector<int>> &samples) {
	if (same_class(samples)) {
		make_leaf(samples, true);
	} else {
		vector<int> dimensions = random_dimensions(samples[0].size());
		pair<int, int> best_split = find_best_split(samples, dimensions);

		if (best_split.first == -1 && best_split.second == -1) {
			make_leaf(samples, false);
		} else {
			pair<vector<vector<int>>, vector<vector<int>>> children;
			children = split(samples, best_split.first, best_split.second);
			make_decision_node(best_split.first, best_split.second);

			left = make_shared<Node>(Node());
			right = make_shared<Node>(Node());

			left->train(children.first);
			right->train(children.second);
		}
	}
}

/*
 * Intoarce rezultatul prezis de catre decision tree
 */
int Node::predict(const vector<int> &image) const {
	if (is_leaf) {
		return result;
	}

	if (image[split_index - 1] <= split_value) {
		return left->predict(image);
	} else {
		return right->predict(image);
	}
}

/*
 * Verifica daca testele primite ca argument au toate aceeasi
 * clasa(rezultat). Este folosit in train pentru a determina daca
 * mai are rost sa caute split-uri.
 * Se retine clasa primului sample
 * si se compara cu celelalte clase
 */
bool same_class(const vector<vector<int>> &samples) {
	int first_class = samples[0][0];
	int size = samples.size();

	for (int i = 1; i < size; ++i) {
		if (samples[i][0] != first_class) {
			return false;
		}
	}

	return true;
}

/*
 * Intoarce entropia testelor primite
 */	
float get_entropy(const vector<vector<int>> &samples) {
	assert(!samples.empty());
	vector<int> indexes;

	int size = samples.size();
	for (int i = 0; i < size; i++) indexes.push_back(i);

	return get_entropy_by_indexes(samples, indexes);
}

/*
 * Intoarce entropia subsetului din setul de teste total(samples)
 * Cu conditia ca subsetul sa contina testele ale caror indecsi se gasesc in
 * vectorul index (Se considera doar liniile din vectorul index)
 */
float get_entropy_by_indexes(const vector<vector<int>> &samples,
							 const vector<int> &index) {
	vector<int> frequency(10, 0);
	int size = index.size();
	float H = 0;

	/*
	 * Se calculeaza numarul de aparitii al fiecarei cifre
	 * in subsetul reprezentat prin valorile din vectorul index
	 */
	for (int i = 0; i < size; ++i) {
		++frequency[samples[index[i]][0]];
	}

	for (int i = 0; i < 10; ++i) {
		if (frequency[i]) {
			/*
			 * Se calculeaza probabilitatea
			 * ca un test sa aiba rezultatul i
			 */
			float p_i = (float)frequency[i] / size;

			/*
			 * Se calculeaza entropia
			*/
			H += (p_i * log2(p_i));
		}
	}

	return -H;
}

/*
 * Intoarce toate valorile (se elimina duplicatele)
 * care apar in setul de teste, pe coloana col
 */
vector<int> compute_unique(const vector<vector<int>> &samples, const int col) {
	vector<int> uniqueValues;
	vector<int> frequency(256, 0);
	int size = samples.size();

	/*
	 * Se marcheaza cu 1 aparitia fiecarei valori
	 */
	for (int i = 0; i < size; ++i) {
		frequency[samples[i][col]] = 1;
	}

	/*
	 * Se retin valorile care au aparut
	 */
	for (int i = 0; i < 256; ++i) {
		if (frequency[i]) {
			uniqueValues.push_back(i);
		}
	}

	return uniqueValues;
}

/*
 * Intoarce cele 2 subseturi de teste obtinute in urma separarii
 * In functie de split_index si split_value
 */
pair<vector<vector<int>>, vector<vector<int>>> split(
	const vector<vector<int>> &samples, const int split_index,
	const int split_value) {
	vector<vector<int>> left, right;

	auto p = get_split_as_indexes(samples, split_index, split_value);
	for (const auto &i : p.first) left.push_back(samples[i]);
	for (const auto &i : p.second) right.push_back(samples[i]);

	return pair<vector<vector<int>>, vector<vector<int>>>(left, right);
}

/*
 * Intoarce indecsii sample-urilor din cele 2 subseturi obtinute in urma
 * separarii in functie de split_index si split_value
 */
pair<vector<int>, vector<int>> get_split_as_indexes(
	const vector<vector<int>> &samples, const int split_index,
	const int split_value) {
	vector<int> left, right;

	int size = samples.size();
	for (int i = 0; i < size; i++) {
		if (samples[i][split_index] <= split_value) {
			left.push_back(i);
		} else {
			right.push_back(i);
		}
	}

	return make_pair(left, right);
}

vector<int> random_dimensions(const int size) {
	/*
	 * Intoarce sqrt(size) dimensiuni diferite pe care sa caute splitul maxim
	 * Precizare: Dimensiunile gasite sunt > 0 si < size
	 */
	int dim = floor(sqrt(size));
	vector<int> rez;
	vector<int> frequency(size, 0);
	std::uniform_int_distribution<int> d(0, size - 1);
	std::random_device rd;

	/*
	 * Se genereaza un numar random
	 */
	for (int i = 0; i < dim; ++i) {
		int value = d(rd);

		/*
		 * Daca valoarea este diferita de 0
		 * si nu a mai fost generata anterior,
		 * se retine aparitia acesteia
		 */
		if (value && !frequency[value]) {
			frequency[value] = 1;
			rez.push_back(value);
		} else {
			/*
			 * In caz contrar,
			 * generarea nu este luata in considerare
			 */
			--i;
		}
	}

	return rez;
}
