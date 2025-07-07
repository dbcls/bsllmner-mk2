import { AddOutlined, ArrowDownwardOutlined, ArrowUpwardOutlined, DeleteOutlineOutlined, DownloadOutlined, OpenInNewOutlined } from "@mui/icons-material"
import {
  Checkbox, FormControlLabel, Typography, Link, TextField, Box,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  IconButton,
  Button,
} from "@mui/material"
import { type SxProps } from "@mui/system"
import { useQueryClient } from "@tanstack/react-query"
import { useState } from "react"
import { useController, useFieldArray, useFormContext } from "react-hook-form"

import CodeBlock from "@/components/CodeBlock"
import OurCard from "@/components/OurCard"
import { usePullOllamaModel } from "@/hooks/usePullOllamaModel"
import { type FormValues, type OllamaModels, type Prompt, PROMPT_ROLES } from "@/schema"
import { getErrorChain } from "@/utils"

interface FormCardProps {
  sx?: SxProps
  models: OllamaModels
}

export default function FormCard({ sx, models }: FormCardProps) {
  const { control } = useFormContext<FormValues>()
  const { field: useDefaultBsEntriesField } = useController({
    name: "useDefaultBsEntries",
    control,
    defaultValue: true,
  })
  const { field: bsEntriesField } = useController({
    name: "bsEntries",
    control,
    defaultValue: "",
  })
  const modelNames = models.models.map((model) => model.name)
  const { field: modelField } = useController({
    name: "model",
    control,
    defaultValue: modelNames[0] || "",
  })
  const {
    fields: promptFields,
    append: appendPrompt,
    remove: removePrompt,
    move: movePrompt,
  } = useFieldArray({
    control,
    name: "prompt",
  })

  // model installation form
  const queryClient = useQueryClient()
  const { control: installModelControl, trigger: installModelTrigger } = useFormContext<{ installModel: string }>()
  const { field: installModelField, fieldState: installModelState } = useController({
    name: "installModel",
    control: installModelControl,
    defaultValue: "",
    rules: {
      required: "Model name is required",
      validate: (value) =>
        !modelNames.includes(value) || `"${value}" is already installed`,
    },
  })
  const installModel = installModelField.value
  const { mutate, isPending } = usePullOllamaModel()
  const installButtonLabel = isPending ? "Installing..." : "Install Model"
  const installButtonDisabled = isPending || !!installModelState.error
  const [installResponse, setInstallResponse] = useState<string | null>(null)
  const installModelHandler = async () => {
    const valid = await installModelTrigger("installModel")
    if (!valid) return

    mutate(installModel, {
      onSuccess: () => {
        setInstallResponse(`Model "${installModel}" installed successfully.`)
        installModelField.onChange("")
        queryClient.invalidateQueries({ queryKey: ["ollamaModels"] })
      },
      onError: (err) => {
        const chain = getErrorChain(err)
        const errorMessages = chain.map((e) => e.message).join("\nCaused by: ")

        setInstallResponse(`Error installing model "${installModel}": ${errorMessages}`)
      },
    })
  }

  return (
    <OurCard sx={sx}>
      <Box>
        <Typography component="h1" sx={{ fontSize: "1.5rem" }}>
          {"BioSample Entries"}
        </Typography>
        <Box sx={{ mx: "1.5rem", mt: "0.5rem" }}>
          <FormControlLabel
            control={
              <Checkbox
                {...useDefaultBsEntriesField}
                checked={useDefaultBsEntriesField.value}
                onChange={(e) => useDefaultBsEntriesField.onChange(e.target.checked)}
              />
            }
            label={<>
              {"Use Default BioSample Entries ("}
              <Link
                href={`${BSLLMNER2_API_URL}/default-bs-entries.json`}
                target="_blank"
                rel="noopener noreferrer"
                underline="hover"
                sx={{ display: "inline-flex", alignItems: "center", gap: "0.25rem" }}
              >
                {"open default json in new tab"}
                <OpenInNewOutlined sx={{ fontSize: "1rem" }} />
              </Link>
              {")"}
            </>}
          />
          {!useDefaultBsEntriesField.value && (<>
            <TextField
              {...bsEntriesField}
              value={bsEntriesField.value ?? ""}
              onChange={(e) => bsEntriesField.onChange(e.target.value)}
              placeholder="Enter BioSample entries in JSON format"
              multiline
              minRows={5}
              maxRows={10}
              fullWidth
              sx={{ mt: "0.5rem" }}
              slotProps={{
                input: {
                  sx: { fontFamily: "monospace" },
                },
              }}
            />
          </>)
          }
        </Box>
      </Box>

      <Box sx={{ mt: "1.5rem" }}>
        <Typography component="h1" sx={{ fontSize: "1.5rem" }}>
          {"Model"}
        </Typography>
        <Box sx={{ mx: "1.5rem", mt: "0.5rem" }}>
          <FormControl fullWidth sx={{ mt: "1rem" }}>
            <InputLabel id="model-select-label">
              {"Model"}
            </InputLabel>
            <Select
              labelId="model-select-label"
              id="model-select"
              label="Model"
              {...modelField}
              value={modelField.value}
              onChange={(e) => modelField.onChange(e.target.value)}
              sx={{ maxWidth: "20rem" }}
            >
              {modelNames.map((name) => (
                <MenuItem key={name} value={name}>
                  {name}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
          <Box sx={{ mt: "1rem", display: "flex", flexDirection: "column", gap: "0.5rem" }}>
            <Typography>
              {"To add a new model, please select from the "}
              <Link
                href={"https://ollama.com/library"}
                target="_blank"
                rel="noopener noreferrer"
                underline="hover"
                sx={{ display: "inline-flex", alignItems: "center", gap: "0.25rem" }}
              >
                {"Ollama Library"}
                <OpenInNewOutlined sx={{ fontSize: "1rem" }} />
              </Link>
              {" and input the model name here."}
            </Typography>
            <Box sx={{ display: "flex", alignItems: "center", gap: "1.5rem" }}>
              <TextField
                {...installModelField}
                error={!!installModelState.error}
                helperText={installModelState.error?.message}
                placeholder="Enter model name to install"
                fullWidth
                sx={{ maxWidth: "20rem" }}
              />
              <Button
                variant="outlined"
                sx={{
                  textTransform: "none",
                  mb: installModelState.error ? "1.4rem" : "0",
                  width: "10rem",
                }}
                onClick={installModelHandler}
                startIcon={<DownloadOutlined />}
                disabled={installButtonDisabled}
              >
                {installButtonLabel}
              </Button>
            </Box>
            <Typography sx={{ fontWeight: "bold" }}>
              {"Install Log:"}
            </Typography>
            <CodeBlock
              content={installResponse ?? "No installation response yet."}
            />
          </Box>
        </Box>
      </Box>

      <Box sx={{ mt: "1.5rem" }}>
        <Typography component="h1" sx={{ fontSize: "1.5rem" }}>
          {"Prompts"}
        </Typography>
        <Box sx={{ mx: "1.5rem", mt: "1rem" }}>
          {promptFields.map((field, index) => (
            <Box
              key={field.id}
              sx={{
                display: "flex",
                flexDirection: "column",
                gap: "0.5rem",
                border: "1px solid #ccc",
                borderRadius: 2,
                p: "1rem",
                mb: "1rem",
              }}
            >
              <Box sx={{ display: "flex", gap: "0.5rem", alignItems: "center" }}>
                <FormControl fullWidth>
                  <InputLabel id={`prompt-${index}-role-label`}>Role</InputLabel>
                  <Select
                    labelId={`prompt-${index}-role-label`}
                    id={`prompt-${index}-role`}
                    label="Role"
                    {...control.register(`prompt.${index}.role` as const)}
                    defaultValue={field.role}
                    sx={{ maxWidth: "20rem" }}
                  >
                    {PROMPT_ROLES.map((role) => (
                      <MenuItem key={role} value={role}>
                        {role}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
                <IconButton
                  onClick={() => index > 0 && movePrompt(index, index - 1)}
                  disabled={index === 0}
                  sx={{ width: "36px", height: "36px", alignSelf: "center" }}
                >
                  <ArrowUpwardOutlined />
                </IconButton>
                <IconButton
                  onClick={() => index < promptFields.length - 1 && movePrompt(index, index + 1)}
                  disabled={index === promptFields.length - 1}
                  sx={{ width: "36px", height: "36px", alignSelf: "center" }}
                >
                  <ArrowDownwardOutlined />
                </IconButton>
                <IconButton
                  onClick={() => removePrompt(index)}
                  sx={{ width: "36px", height: "36px", alignSelf: "center", mr: "0.5rem" }}
                >
                  <DeleteOutlineOutlined />
                </IconButton>
              </Box>
              <TextField
                multiline
                minRows={2}
                maxRows={5}
                fullWidth
                placeholder="Enter prompt content"
                {...control.register(`prompt.${index}.content` as const)}
                defaultValue={field.content}
              />
            </Box>
          ))}
          <Typography sx={{ fontWeight: "bold", my: "1rem" }}>
            {"NOTE: The last prompt will be extended by concatenating the JSON content of each BioSample Entry."}
          </Typography>
          <Button
            variant="outlined"
            onClick={() =>
              appendPrompt({ role: "user", content: "" } satisfies Prompt)
            }
            sx={{ textTransform: "none" }}
            startIcon={<AddOutlined />}
          >
            {"Add Prompt"}
          </Button>
        </Box>
      </Box>
    </OurCard >
  )
}
