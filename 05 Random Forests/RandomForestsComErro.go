package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

type Node struct {
	Feature   int
	Threshold float64
	Left      *Node
	Right     *Node
	Value     interface{}
}

func NewNode(feature int, threshold float64, left, right *Node, value interface{}) *Node {
	return &Node{
		Feature:   feature,
		Threshold: threshold,
		Left:      left,
		Right:     right,
		Value:     value,
	}
}

func (node *Node) IsLeafNode() bool {
	return node.Value != nil
}

type DecisionTree struct {
	MinSamplesSplit int
	MaxDepth        int
	NFeatures       int
	Root            *Node
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
	dt.Root = dt.growTree(X, y)
}

func (dt *DecisionTree) growTree(X [][]float64, y []int, depth int) *Node {
	nSamples, nFeatures := len(X), len(X[0])
	nLabels := len(uniqueIntSlice(y))

	if depth >= dt.MaxDepth || nLabels == 1 || nSamples < dt.MinSamplesSplit {
		leafValue := mostCommonLabel(y)
		return NewNode(0, 0.0, nil, nil, leafValue)
	}

	featIdxs := randomSample(nFeatures, dt.NFeatures)

	bestFeature, bestThreshold := dt.bestSplit(X, y, featIdxs)

	leftIdxs, rightIdxs := splitIndexes(getColumn(X, bestFeature), bestThreshold)
	left := dt.growTree(subset(X, leftIdxs), subsetInt(y, leftIdxs), depth+1)
	right := dt.growTree(subset(X, rightIdxs), subsetInt(y, rightIdxs), depth+1)

	return NewNode(bestFeature, bestThreshold, left, right, nil)
}

func (dt *DecisionTree) bestSplit(X [][]float64, y []int, featIdxs []int) (int, float64) {
	bestGain := -1.0
	var splitIdx int
	var splitThreshold float64

	for _, featIdx := range featIdxs {
		XColumn := getColumn(X, featIdx)
		thresholds := uniqueFloatSlice(XColumn)

		for _, threshold := range thresholds {
			gain := dt.informationGain(y, XColumn, threshold)

			if gain > bestGain {
				bestGain = gain
				splitIdx = featIdx
				splitThreshold = threshold
			}
		}
	}

	return splitIdx, splitThreshold
}

func (dt *DecisionTree) informationGain(y []int, XColumn []float64, threshold float64) float64 {
	parentEntropy := entropy(y)

	leftIdxs, rightIdxs := splitIndexes(XColumn, threshold)
	leftEntropy := entropy(subsetInt(y, leftIdxs))
	rightEntropy := entropy(subsetInt(y, rightIdxs))

	n := float64(len(y))
	nLeft, nRight := float64(len(leftIdxs)), float64(len(rightIdxs))

	childEntropy := (nLeft/n) * leftEntropy + (nRight/n) * rightEntropy

	return 	parentEntropy - childEntropy
}

func splitIndexes(X []float64, threshold float64) ([]int, []int) {
	var leftIdxs, rightIdxs []int

	for i, val := range X {
		if val <= threshold {
			leftIdxs = append(leftIdxs, i)
		} else {
			rightIdxs = append(rightIdxs, i)
		}
	}

	return leftIdxs, rightIdxs
}

func getColumn(X [][]float64, idx int) []float64 {
	column := make([]float64, len(X))

	for i, row := range X {
		column[i] = row[idx]
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

func subsetInt(arr []int, idxs []int) []int {
	subset := make([]int, len(idxs))

	for i, idx := range idxs {
		subset[i] = arr[idx]
	}

	return subset
}

func uniqueIntSlice(arr []int) []int {
	uniqueMap := make(map[int]bool)

	for _, item := range arr {
		uniqueMap[item] = true
	}

	uniqueSlice := make([]int, 0, len(uniqueMap))

	for item := range uniqueMap {
		uniqueSlice = append(uniqueSlice, item)
	}

	return uniqueSlice
}

func uniqueFloatSlice(arr []float64) []float64 {
	uniqueMap := make(map[float64]bool)

	for _, item := range arr {
		uniqueMap[item] = true
	}

	uniqueSlice := make([]float64, 0, len(uniqueMap))

	for item := range uniqueMap {
		uniqueSlice = append(uniqueSlice, item)
	}

	return uniqueSlice
}

func entropy(y []int) float64 {
	counter := make(map[int]int)

	for _, val := range y {
		counter[val]++
	}

	entropy := 0.0
	n := float64(len(y))

	for _, count := range counter {
		probability := float64(count) / n
		entropy -= probability * math.Log2(probability)
	}

	return entropy
}

func mostCommonLabel(y []int) int {
	counter := make(map[int]int)

	for _, val := range y {
		counter[val]++
	}

	maxCount := 0
	mostCommon := 0

	for val, count := range counter {
		if count > maxCount {
			maxCount = count
			mostCommon = val
		}
	}

	return mostCommon
}

type RandomForest struct {
	NTrees            int
	MaxDepth          int
	MinSamplesSplit   int
	NFeatures         int
	Trees             []*DecisionTree
}

func NewRandomForest(nTrees, maxDepth, minSamplesSplit, nFeatures int) *RandomForest {
	return &RandomForest{
		NTrees:            nTrees,
		MaxDepth:          maxDepth,
		MinSamplesSplit:   minSamplesSplit,
		NFeatures:         nFeatures,
		Trees:             make([]*DecisionTree, nTrees),
	}
}

func (rf *RandomForest) Fit(X [][]float64, y []int) {
	rand.Seed(time.Now().UnixNano())

	for i := 0; i < rf.NTrees; i++ {
		tree := NewDecisionTree(rf.MinSamplesSplit, rf.MaxDepth, rf.NFeatures)
		XSample, ySample := rf.bootstrapSamples(X, y)
		tree.Fit(XSample, ySample)
		rf.Trees[i] = tree
	}
}

func (rf *RandomForest) bootstrapSamples(X [][]float64, y []int) ([][]float64, []int) {
	nSamples := len(X)
	idxs := make([]int, nSamples)

	for i := 0; i < nSamples; i++ {
		idxs[i] = rand.Intn(nSamples)
	}

	XSample := make([][]float64, nSamples)
	ySample := make([]int, nSamples)

	for i, idx := range idxs {
		XSample[i] = X[idx]
		ySample[i] = y[idx]
	}

	return XSample, ySample
}

func (rf *RandomForest) predict(X [][]float64) []int {
	nSamples := len(X)
	predictions := make([]int, nSamples)

	for i := 0; i < nSamples; i++ {
		predictions[i] = rf.predictSingle(X[i])
	}

	return predictions
}

func (rf *RandomForest) predictSingle(x []float64) int {
	predictionMap := make(map[int]int)

	for _, tree := range rf.Trees {
		prediction := rf.traverseTree(x, tree.Root)
		predictionMap[prediction]++
	}

	maxCount := 0
	mostCommon := 0

	for val, count := range predictionMap {
		if count > maxCount {
			maxCount = count
			mostCommon = val
		}
	}

	return mostCommon
}

func (rf *RandomForest) traverseTree(x []float64, node *Node) int {
	if node.IsLeafNode() {
		value, _ := node.Value.(int)
		return value
	}

	if x[node.Feature] <= node.Threshold {
		return rf.traverseTree(x, node.Left)
	}

	return rf.traverseTree(x, node.Right)
}

func main() {
	// Example usage
	X := [][]float64{
		{1.0, 2.0, 3.0},
		{4.0, 5.0, 6.0},
		{7.0, 8.0, 9.0},
	}
	y := []int{0, 1, 0}

	rf := NewRandomForest(10, 10, 2, 2)
	rf.Fit(X, y)

	newX := [][]float64{
		{2.0, 3.0},
		{5.0, 6.0},
		{8.0, 9.0},
	}
	predictions := rf.predict(newX)

	fmt.Println(predictions)
}

