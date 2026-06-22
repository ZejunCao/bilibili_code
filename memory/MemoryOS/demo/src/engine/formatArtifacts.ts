import { getStepGuide } from '../data/stepGuides'

function formatScalar(value: unknown): string {
  return JSON.stringify(value)
}

/** 嵌套结构不附加注释，仅格式化 JSON */
function formatNested(value: unknown, indent: number): string {
  const sp = '  '.repeat(indent)
  if (value === null || value === undefined) return 'null'
  if (typeof value !== 'object') return formatScalar(value)

  if (Array.isArray(value)) {
    if (value.length === 0) return '[]'
    const lines = value.map((item) => {
      const inner = formatNested(item, indent + 1)
      if (inner.includes('\n')) {
        const padded = inner
          .split('\n')
          .map((line, i) => (i === 0 ? line : `${sp}  ${line}`))
          .join('\n')
        return `${sp}  ${padded}`
      }
      return `${sp}  ${inner}`
    })
    return `[\n${lines.join(',\n')}\n${sp}]`
  }

  const obj = value as Record<string, unknown>
  const keys = Object.keys(obj)
  if (keys.length === 0) return '{}'
  const lines = keys.map((key) => {
    const inner = formatNested(obj[key], indent + 1)
    if (inner.includes('\n')) {
      return `${sp}  ${JSON.stringify(key)}: ${inner}`
    }
    return `${sp}  ${JSON.stringify(key)}: ${inner}`
  })
  return `{\n${lines.join(',\n')}\n${sp}}`
}

function formatSessionObjLine(
  key: string,
  value: unknown,
  comments: Record<string, string>,
): string {
  const commentSuffix = comments[key] ? `  // ${comments[key]}` : ''
  // w4f 与 mid_term.add_session 一致：details 即 processed_details，展示不重复展开
  if (key === 'details') {
    return `    ${JSON.stringify(key)}: processed_details${commentSuffix}`
  }
  const inner = formatNested(value, 2)
  if (inner.includes('\n')) {
    const [first, ...rest] = inner.split('\n')
    return `    ${JSON.stringify(key)}: ${first}${commentSuffix}\n${rest.join('\n')}`
  }
  return `    ${JSON.stringify(key)}: ${inner}${commentSuffix}`
}

function formatW4fArtifacts(
  data: { processed_details: unknown; session_obj: Record<string, unknown> },
  comments: Record<string, string>,
): string {
  const pd = formatNested(data.processed_details, 1)
  const pdComment = comments.processed_details ? `  // ${comments.processed_details}` : ''
  let pdLine: string
  if (pd.includes('\n')) {
    const [first, ...rest] = pd.split('\n')
    pdLine = `  "processed_details": ${first}${pdComment}\n${rest.join('\n')}`
  } else {
    pdLine = `  "processed_details": ${pd}${pdComment}`
  }

  const sessionLines = Object.keys(data.session_obj).map((key) =>
    formatSessionObjLine(key, data.session_obj[key], comments),
  )
  const sessionComment = comments.session_obj ? `  // ${comments.session_obj}` : ''
  return `{
${pdLine},
  "session_obj": {${sessionComment}
${sessionLines.join(',\n')}
  }
}`
}

/** 将 artifacts 格式化为 JSON 风格文本；注释仅出现在最外层字段行 */
export function formatArtifactsWithComments(stepId: string, data: unknown): string {
  if (data === null || data === undefined) return ''
  const comments = getStepGuide(stepId).artifactFieldComments
  if (typeof data !== 'object' || Array.isArray(data)) {
    return formatNested(data, 0)
  }

  const obj = data as Record<string, unknown>
  if (stepId === 'w4f' && obj.processed_details != null && obj.session_obj != null) {
    return formatW4fArtifacts(
      obj as { processed_details: unknown; session_obj: Record<string, unknown> },
      comments,
    )
  }

  const keys = Object.keys(obj)
  if (keys.length === 0) return '{}'

  const lines = keys.map((key) => {
    const inner = formatNested(obj[key], 1)
    const commentSuffix = comments[key] ? `  // ${comments[key]}` : ''
    if (inner.includes('\n')) {
      const [first, ...rest] = inner.split('\n')
      return `  ${JSON.stringify(key)}: ${first}${commentSuffix}\n${rest.join('\n')}`
    }
    return `  ${JSON.stringify(key)}: ${inner}${commentSuffix}`
  })
  return `{\n${lines.join(',\n')}\n}`
}
