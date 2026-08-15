export interface ContainedMediaRect {
  x: number
  y: number
  width: number
  height: number
}

/** Rectangle occupied by object-contain media inside its element. */
export function containedMediaRect(
  boxWidth: number,
  boxHeight: number,
  mediaWidth: number,
  mediaHeight: number,
): ContainedMediaRect {
  if (boxWidth <= 0 || boxHeight <= 0 || mediaWidth <= 0 || mediaHeight <= 0) {
    return { x: 0, y: 0, width: boxWidth, height: boxHeight }
  }
  const scale = Math.min(boxWidth / mediaWidth, boxHeight / mediaHeight)
  const width = mediaWidth * scale
  const height = mediaHeight * scale
  return {
    x: (boxWidth - width) / 2,
    y: (boxHeight - height) / 2,
    width,
    height,
  }
}
