package main

import (
	"fmt"
	"math"
	"sort"
)

type KNN struct {
	k       int
	X_train [][]float64
	y_train []string
}

func euclideanDistance(x1, x2 []float64) float64 {
	var distance float64
	for i := 0; i < len(x1); i++ {
		distance += math.Pow(x1[i]-x2[i], 2)
	}
	return math.Sqrt(distance)
}

func (knn *KNN) Fit(X [][]float64, y []string) {
	knn.X_train = X
	knn.y_train = y
}

func (knn *KNN) Predict(X [][]float64) []string {
	var predictions []string
	for _, x := range X {
		prediction := knn.predict(x)
		predictions = append(predictions, prediction)
	}
	return predictions
}

func (knn *KNN) predict(x []float64) string {
	var distances []float64
	for _, x_train := range knn.X_train {
		distance := euclideanDistance(x, x_train)
		distances = append(distances, distance)
	}

	k_indices := getKIndices(distances, knn.k)
	k_nearest_labels := getKNearestLabels(k_indices, knn.y_train)

	most_common := getMostCommon(k_nearest_labels)
	return most_common
}

func getKIndices(distances []float64, k int) []int {
	type DistanceIndex struct {
		distance float64
		index    int
	}

	var sortedDistances []DistanceIndex
	for i, distance := range distances {
		sortedDistances = append(sortedDistances, DistanceIndex{distance, i})
	}

	sort.Slice(sortedDistances, func(i, j int) bool {
		return sortedDistances[i].distance < sortedDistances[j].distance
	})

	var kIndices []int
	for i := 0; i < k; i++ {
		kIndices = append(kIndices, sortedDistances[i].index)
	}

	return kIndices
}

func getKNearestLabels(indices []int, y_train []string) []string {
	var kNearestLabels []string
	for _, index := range indices {
		kNearestLabels = append(kNearestLabels, y_train[index])
	}
	return kNearestLabels
}

func getMostCommon(labels []string) string {
	counts := make(map[string]int)
	for _, label := range labels {
		counts[label]++
	}

	var mostCommonLabel string
	maxCount := 0
	for label, count := range counts {
		if count > maxCount {
			mostCommonLabel = label
			maxCount = count
		}
	}

	return mostCommonLabel
}

func main() {
	X_train := [][]float64{{1, 1}, {1, 2}, {2, 2}, {4, 4}, {4, 5}, {5, 5}}
	y_train := []string{"A", "A", "A", "B", "B", "B"}

	knn := KNN{k: 3}
	knn.Fit(X_train, y_train)

	X_test := [][]float64{{3, 3}, {5, 6}, {1, 2}}
	predictions := knn.Predict(X_test)

	fmt.Println(predictions)
}
