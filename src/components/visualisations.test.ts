import { test, expect } from "bun:test";
import { DEFAULT_VISUALISATION, VISUALISATIONS, visualisation } from "./visualisations";

test("there are at least five styles, each with its own id", () => {
  expect(VISUALISATIONS.length).toBeGreaterThanOrEqual(5);
  expect(new Set(VISUALISATIONS.map((v) => v.id)).size).toBe(VISUALISATIONS.length);
});

test("the default is one of them and an unknown name falls back to it", () => {
  expect(visualisation(DEFAULT_VISUALISATION).id).toBe(DEFAULT_VISUALISATION);
  expect(visualisation("no-such-style").id).toBe(DEFAULT_VISUALISATION);
});

test("the ids match what Rust accepts", () => {
  // settings.rs VISUALISATIONS. A new style has to be added in both places.
  expect(VISUALISATIONS.map((v) => v.id)).toEqual(["waterfall", "bars", "mirror", "ring", "dots", "curve"]);
});
