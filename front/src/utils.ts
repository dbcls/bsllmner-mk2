interface ErrorWithCause extends Error {
  cause?: unknown
}

export const getErrorChain = (error: unknown): ErrorWithCause[] => {
  const errors: ErrorWithCause[] = []
  let currentError: unknown = error
  while (currentError && typeof currentError === "object" && "message" in currentError) {
    errors.push(currentError as ErrorWithCause)
    currentError = (currentError as ErrorWithCause).cause
  }

  return errors
}

export const getNowStr = (date: Date = new Date()): string => {
  const pad = (num: number): string => num.toString().padStart(2, "0")
  const YYYY = date.getFullYear()
  const MM = pad(date.getMonth() + 1)
  const DD = pad(date.getDate())
  const hh = pad(date.getHours())
  const mm = pad(date.getMinutes())
  const ss = pad(date.getSeconds())

  return `${YYYY}${MM}${DD}_${hh}${mm}${ss}`
}

export const parseNowStr = (nowStr: string): Date => {
  const YYYY = parseInt(nowStr.slice(0, 4), 10)
  const MM = parseInt(nowStr.slice(4, 6), 10) - 1 // Months are zero-based
  const DD = parseInt(nowStr.slice(6, 8), 10)
  const hh = parseInt(nowStr.slice(9, 11), 10)
  const mm = parseInt(nowStr.slice(11, 13), 10)
  const ss = parseInt(nowStr.slice(13, 15), 10)

  return new Date(YYYY, MM, DD, hh, mm, ss)
}
