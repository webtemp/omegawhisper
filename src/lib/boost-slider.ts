// The boost slider runs from 0.5x to 100x, spread by ratio so 1, 2, 5, 10,
// 20, 50 and 100 sit at even distances. Values are kept to a tenth.
export const BOOST_MIN = 0.5;
export const BOOST_MAX = 100;
export const SLIDER_STEPS = 1000;

export function boostToSlider(boost: number): number {
  const clamped = Math.min(BOOST_MAX, Math.max(BOOST_MIN, boost));
  const ratio = Math.log(clamped / BOOST_MIN) / Math.log(BOOST_MAX / BOOST_MIN);
  return Math.round(ratio * SLIDER_STEPS);
}

export function sliderToBoost(position: number): number {
  const boost = BOOST_MIN * Math.pow(BOOST_MAX / BOOST_MIN, position / SLIDER_STEPS);
  return Math.round(boost * 10) / 10;
}
