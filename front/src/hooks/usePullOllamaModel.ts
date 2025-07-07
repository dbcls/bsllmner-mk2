import { useMutation } from "@tanstack/react-query"

const pullOllamaModel = async (modelName: string): Promise<string> => {
  try {
    const res = await fetch(`${BSLLMNER2_OLLAMA_URL}/api/pull`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: modelName,
        stream: false,
      }),
    })
    if (!res.ok) {
      throw new Error(`HTTP Error: ${res.status} ${res.statusText}`)
    }
    const data = await res.json()
    return JSON.stringify(data, null, 2)
  } catch (error) {
    throw new Error("Failed to pull Ollama model", { cause: error })
  }
}

export const usePullOllamaModel = () =>
  useMutation<string, Error, string>({
    mutationFn: pullOllamaModel,
  })
