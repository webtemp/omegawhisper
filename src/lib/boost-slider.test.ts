import { test, expect } from "bun:test";
import { boostToSlider, sliderToBoost, SLIDER_STEPS } from "./boost-slider";

test("the ends of the slider are 0.5x and 100x", () => {
  expect(sliderToBoost(0)).toBe(0.5);
  expect(sliderToBoost(SLIDER_STEPS)).toBe(100);
  expect(boostToSlider(0.5)).toBe(0);
  expect(boostToSlider(100)).toBe(SLIDER_STEPS);
});

test("1x sits well inside the slider, not at the left edge", () => {
  const off = boostToSlider(1);
  expect(off).toBeGreaterThan(SLIDER_STEPS * 0.1);
  expect(off).toBeLessThan(SLIDER_STEPS * 0.2);
});

test("every tenth from 0.5 to 100 survives a trip through the slider", () => {
  for (const boost of [0.5, 0.7, 1, 1.5, 2, 4, 10, 32.1, 50, 99.9, 100]) {
    expect(sliderToBoost(boostToSlider(boost))).toBeCloseTo(boost, 0);
  }
});

test("a value outside the range is pulled inside", () => {
  expect(boostToSlider(0)).toBe(0);
  expect(boostToSlider(500)).toBe(SLIDER_STEPS);
});
