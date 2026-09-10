import assert from "node:assert/strict";
import test from "node:test";
import { candidateFeatures, heuristicScores, learnedScores, validatePolicy } from "../ranker.js";
import { policy } from "./fixtures.js";

const statistics = { n_recipes: 4, ingredient_frequency: [4, 2, 1, 0] };

test("feature kernel uses set semantics and explicit unknown times", () => {
  const [exact, partial] = candidateFeatures([0, 1, 1], [[1, 0, 0], [1, 2]], statistics,
    { totalMinutes: [null, 10], maxTotalMinutes: 20 });
  assert.equal(exact.length, 20);
  assert.equal(exact[12], 1);
  assert.equal(exact[13], 1);
  assert.equal(exact[14], 0);
  assert.equal(exact[15], 1);
  assert.deepEqual([...exact.slice(16)], [0, 0, 0, 0]);
  assert.equal(partial[4], 0.5);
  assert.equal(partial[17], 0.5);
  assert.equal(partial[18], 0.5);
  assert.equal(partial[19], 1);
  assert.deepEqual(candidateFeatures([0], [], statistics), []);
});

test("known zero time is different from unknown and feature budgets must be positive", () => {
  const [zero, unknown] = candidateFeatures([0], [[0], [1]], statistics,
    { totalMinutes: [0, null], maxTotalMinutes: 1 });
  assert.deepEqual([...zero.slice(14)], [1, 1, 0, 0, 1, 1]);
  assert.deepEqual([...unknown.slice(14)], [0, 1, 0, 0, 0, 0]);
  assert.throws(() => candidateFeatures([0], [[0]], statistics, { maxTotalMinutes: 0 }));
});

test("invalid IDs, statistics and time values are rejected rather than guessed", () => {
  for (const available of [[], [true], [-1], [4], ["1"]]) {
    assert.throws(() => candidateFeatures(available, [[0]], statistics));
  }
  for (const counts of [[0.5], [5], [NaN], [-1], [true], []]) {
    assert.throws(() => candidateFeatures([0], [[0]], { n_recipes: 4, ingredient_frequency: counts }));
  }
  for (const totalMinutes of [[NaN], [false], ["10"], [], [-1]]) {
    assert.throws(() => candidateFeatures([0], [[0]], statistics, { totalMinutes }));
  }
});

test("the MLP computes a real tanh forward pass and validates its feature mask", () => {
  const rows = candidateFeatures([0], [[0], [1]], statistics);
  const model = policy();
  validatePolicy(model);
  assert.deepEqual(learnedScores(rows, model), [Math.fround(Math.tanh(1)), 0]);
  model.tensors.feature_mask[0] = 0;
  assert.throws(() => validatePolicy(model));
  model.tensors.feature_mask[0] = 1;
  model.tensors["network.0.weight"][0][0] = Infinity;
  assert.throws(() => validatePolicy(model));
});

test("baseline and feature validation match their documented roles", () => {
  const rows = candidateFeatures([0], [[0]], statistics);
  assert.ok(Math.abs(heuristicScores(rows)[0] - 5.6) < 1e-12);
  assert.throws(() => heuristicScores([new Array(20).fill(NaN)]));
  assert.throws(() => learnedScores([new Array(19).fill(0)], policy()));
});
