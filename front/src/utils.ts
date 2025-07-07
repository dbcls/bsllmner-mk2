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
