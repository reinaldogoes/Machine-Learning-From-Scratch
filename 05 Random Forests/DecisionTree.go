package main

import (
	"fmt"
	"math"
	"math/rand"
)

type Node struct {
	Feature   int
	Threshold float64
	Left      *Node
	Right     *Node
	Value     interface{}
}

func (n *Node) IsLeafNode() bool {
	return n.Value != nil
}

type DecisionTree struct {
	MinSamplesSplit int
	MaxDepth        int
	NFeatures       int
	Root            *Node
}

func splitIndexes(XColumn []float64, threshold float64) ([]int, []int) {
	leftIdxs := make([]int, 0)
	rightIdxs := make([]int, 0)

	for i, val := range XColumn {
		if val <= threshold {
			leftIdxs = append(leftIdxs, i)
		} else {
			rightIdxs = append(rightIdxs, i)
		}
	}

	return leftIdxs, rightIdxs
}

func NewDecisionTree(minSamplesSplit, maxDepth, nFeatures int) *DecisionTree {
	return &DecisionTree{
		MinSamplesSplit: minSamplesSplit,
		MaxDepth:        maxDepth,
		NFeatures:       nFeatures,
		Root:            nil,
	}
}

func (dt *DecisionTree) Fit(X [][]float64, y []int) {
	if dt.NFeatures == 0 {
		dt.NFeatures = len(X[0])
	}
	dt.Root = dt.growTree(X, y)
}

func (dt *DecisionTree) growTree(X [][]float64, y []int) *Node {
	nSamples := len(X)
	nFeats := len(X[0])
	nLabels := len(uniqueIntSlice(y))

	if len(y) < dt.MinSamplesSplit || nLabels == 1 || nSamples == 0 || nFeats == 0 {
		leafValue := mostCommonLabel(y)
		return &Node{Value: leafValue}
	}

	featIdxs := randomSample(nFeats, dt.NFeatures)

	bestFeature, bestThreshold := dt.bestSplit(X, y, featIdxs)

	leftIdxs, rightIdxs := dt.split(X, bestFeature, bestThreshold)
	left := dt.growTree(subset(X, leftIdxs), subsetInt(y, leftIdxs))
	right := dt.growTree(subset(X, rightIdxs), subsetInt(y, rightIdxs))

	return &Node{
		Feature:   bestFeature,
		Threshold: bestThreshold,
		Left:      left,
		Right:     right,
	}
}

func (dt *DecisionTree) bestSplit(X [][]float64, y []int, featIdxs []int) (int, float64) {
	bestGain := -1.0
	splitIdx, splitThreshold := -1, -1.0

	for _, featIdx := range featIdxs {
		XColumn := getColumn(X, featIdx)
		thresholds := uniqueFloatSlice(XColumn)

		for _, thr := range thresholds {
			gain := dt.informationGain(y, XColumn, thr)

			if gain > bestGain {
				bestGain = gain
				splitIdx = featIdx
				splitThreshold = thr
			}
		}
	}

	return splitIdx, splitThreshold
}

func (dt *DecisionTree) informationGain(y []int, XColumn []float64, threshold float64) float64 {
	parentEntropy := entropy(y)

	leftIdxs, rightIdxs := splitIndexes(XColumn, threshold)
	if len(leftIdxs) == 0 || len(rightIdxs) == 0 {
		return 0.0
	}

	n := float64(len(y))
	nL, nR := float64(len(leftIdxs)), float64(len(rightIdxs))
	eL, eR := entropy(subsetInt(y, leftIdxs)), entropy(subsetInt(y, rightIdxs))
	childEntropy := (nL/n)*eL + (nR/n)*eR

	informationGain := parentEntropy - childEntropy
	return informationGain
}

func (dt *DecisionTree) split(X [][]float64, feature int, threshold float64) ([]int, []int) {
	leftIdxs := make([]int, 0)
	rightIdxs := make([]int, 0)
	for i, x := range X {
		if x[feature] <= threshold {
			leftIdxs = append(leftIdxs, i)
		} else {
			rightIdxs = append(rightIdxs, i)
		}
	}

	return leftIdxs, rightIdxs

}

func entropy(y []int) float64 {
	hist := make(map[int]int)
	for _, label := range y {
		hist[label]++
	}

	entropy := 0.0
	total := float64(len(y))
	for _, count := range hist {
		probability := float64(count) / total
		entropy -= probability * math.Log2(probability)
	}

	return entropy
}

func mostCommonLabel(y []int) int {
	counter := make(map[int]int)
	maxCount := 0
	mostCommon := 0

	for _, label := range y {
		counter[label]++
		if counter[label] > maxCount {
			maxCount = counter[label]
			mostCommon = label
		}
	}

	return mostCommon
}

func uniqueIntSlice(slice []int) []int {
	uniqueMap := make(map[int]bool)
	uniqueSlice := make([]int, 0)

	for _, item := range slice {
		if !uniqueMap[item] {
			uniqueMap[item] = true
			uniqueSlice = append(uniqueSlice, item)
		}
	}

	return uniqueSlice

}

func uniqueFloatSlice(slice []float64) []float64 {
	uniqueMap := make(map[float64]bool)
	uniqueSlice := make([]float64, 0)

	for _, item := range slice {
		if !uniqueMap[item] {
			uniqueMap[item] = true
			uniqueSlice = append(uniqueSlice, item)
		}
	}

	return uniqueSlice
}

func getColumn(X [][]float64, colIdx int) []float64 {
	column := make([]float64, len(X))

	for i, row := range X {
		column[i] = row[colIdx]
	}

	return column

}

func subset(X [][]float64, idxs []int) [][]float64 {
	subset := make([][]float64, len(idxs))

	for i, idx := range idxs {
		subset[i] = X[idx]
	}

	return subset

}

func subsetInt(slice []int, idxs []int) []int {
	subset := make([]int, len(idxs))
	for i, idx := range idxs {
		subset[i] = slice[idx]
	}

	return subset

}

func randomSample(maxVal, sampleSize int) []int {
	if sampleSize >= maxVal {
		return generateRange(maxVal)
	}

	sample := make([]int, sampleSize)
	indices := generateRange(maxVal)

	for i := 0; i < sampleSize; i++ {
		idx := rand.Intn(maxVal - i)
		sample[i] = indices[idx]
		indices[idx], indices[maxVal-i-1] = indices[maxVal-i-1], indices[idx]
	}

	return sample

}

func generateRange(maxVal int) []int {
	rangeSlice := make([]int, maxVal)

	for i := 0; i < maxVal; i++ {
		rangeSlice[i] = i
	}

	return rangeSlice

}

func main() {
	// Exemplo de uso
	X := [][]float64{
		{1.0, 2.0, 3.0},
		{4.0, 5.0, 6.0},
		{7.0, 8.0, 9.0},
	}

	y := []int{0, 1, 0}

	dt := NewDecisionTree(2, 100, 3)
	dt.Fit(X, y)

	// Exemplo de previsão
	testData := [][]float64{
		{2.0, 4.0, 6.0},
		{1.0, 8.0, 3.0},
	}

	predictions := dt.Predict(testData)
	fmt.Println(predictions)
}

// Predict realiza a previsão para um conjunto de dados de entrada
func (dt *DecisionTree) Predict(X [][]float64) []interface{} {
	predictions := make([]interface{}, len(X))
	for i, x := range X {
		predictions[i] = dt.TraverseTree(x, dt.Root)
	}
	return predictions
}

// TraverseTree percorre a árvore de decisão para fazer a previsão de um único exemplo
func (dt *DecisionTree) TraverseTree(x []float64, node *Node) interface{} {
	if node.IsLeafNode() {
		return node.Value
	}

	if x[node.Feature] <= node.Threshold {
		return dt.TraverseTree(x, node.Left)
	}

	return dt.TraverseTree(x, node.Right)
}

/*
Espero que isso ajude! Certifique-se de incluir as importações necessárias e ajustar o código conforme necessário para atender às suas necessidades.
*/
