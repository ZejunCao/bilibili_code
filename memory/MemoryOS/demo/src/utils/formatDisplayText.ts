/** 演示展示：去掉 tab，并去掉各行公共前导空白，避免 Python 源里的缩进导致错位 */
export function formatDisplayText(text: string): string {
  const noTabs = text.replace(/\t/g, '')
  const lines = noTabs.split('\n')
  const minIndent = lines
    .filter((line) => line.trim().length > 0)
    .reduce((min, line) => {
      const lead = line.match(/^(\s*)/)?.[1].length ?? 0
      return Math.min(min, lead)
    }, Number.POSITIVE_INFINITY)

  if (!Number.isFinite(minIndent) || minIndent <= 0) {
    return noTabs.trimEnd()
  }

  return lines
    .map((line) => (line.length >= minIndent ? line.slice(minIndent) : line))
    .join('\n')
    .trimEnd()
}
