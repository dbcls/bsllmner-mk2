import { useEffect } from "react"
import { FormProvider, useForm } from "react-hook-form"

import FormCard from "@/components/FormCard"
import Frame from "@/components/Frame"
import Loading from "@/components/Loading"
import { useDefaultExtractPrompt } from "@/hooks/useDefaultExtractPrompt"
import { useOllamaModels } from "@/hooks/useOllamaModels"
import { useServiceInfo } from "@/hooks/useServiceInfo"
import { type FormValues, defaultFormValues } from "@/schema"

export default function Home() {
  const serviceInfoQuery = useServiceInfo()
  const ollamaModelsQuery = useOllamaModels()
  const defaultExtractPromptQuery = useDefaultExtractPrompt()
  const loading = serviceInfoQuery.isLoading ||
    ollamaModelsQuery.isLoading ||
    defaultExtractPromptQuery.isLoading

  const methods = useForm<FormValues>({
    defaultValues: defaultFormValues,
    mode: "onBlur",
    reValidateMode: "onBlur",
  })

  // Initialize default values based on the fetched data
  useEffect(() => {
    if (defaultExtractPromptQuery.data) {
      methods.reset({
        ...defaultFormValues,
        prompt: defaultExtractPromptQuery.data,
      })
    }
  }, [methods, defaultExtractPromptQuery.data])

  if (loading) {
    return <Frame><Loading msg="Loading..." /></Frame>
  }

  return (
    <Frame>
      <FormProvider {...methods}>
        <FormCard sx={{ my: "1.5rem" }} models={ollamaModelsQuery.data!} />
      </FormProvider>
    </Frame>
  )
}
