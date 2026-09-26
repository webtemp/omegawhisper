import { test, expect } from "bun:test";
import { LANGUAGES } from "./languages";

test("detection comes first and every code is listed once", () => {
  expect(LANGUAGES[0][0]).toBe("auto");
  expect(new Set(LANGUAGES.map(([code]) => code)).size).toBe(LANGUAGES.length);
});
