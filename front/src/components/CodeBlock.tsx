import { Box, Paper } from "@mui/material"
import { type SxProps } from "@mui/system"

interface CodeBlockProps {
  sx?: SxProps
  content: string
}

export default function CodeBlock({ sx, content }: CodeBlockProps) {
  return (
    <Paper variant="outlined" sx={{ ...sx, p: "0.5rem 1rem" }}>
      <Box sx={{ fontFamily: "monospace", overflowX: "auto" }}>
        <pre>{content}</pre>
      </Box>
    </Paper>
  )
}
