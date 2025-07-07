import { Box, Typography } from "@mui/material"
import type { SxProps } from "@mui/system"

export interface AppFooterProps {
  sx?: SxProps
}

export default function AppFooter({ sx }: AppFooterProps) {
  return (
    <Box component="footer" sx={{ ...sx, margin: "1.5rem 0" }}>
      <Typography variant="body2" align="center" color="text.secondary"
        children={`© 2025-${new Date().getFullYear()} DBCLS.`}
      />
      <Typography variant="body2" align="center" color="text.secondary" sx={{ letterSpacing: "0.1rem" }}
        children={`bsllmner-mk2 ${__APP_VERSION__}`}
      />
    </Box>
  )
}
