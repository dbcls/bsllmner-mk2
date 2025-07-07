import { Box, Paper } from "@mui/material"
import type { SxProps } from "@mui/system"
import React from "react"

export interface OurCardProps {
  sx?: SxProps
  children: React.ReactNode
}

export default function OurCard({ sx, children }: OurCardProps) {
  return (
    <Paper sx={{
      ...sx,
      borderRadius: 2,
      boxShadow: "0px 2px 4px rgba(0, 0, 0, 0.1)",
    }}>
      <Box sx={{ p: "1.5rem 3rem" }}>
        {children}
      </Box>
    </Paper>
  )
}
