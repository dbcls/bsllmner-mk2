import { AppBar, Box, colors, Link as MuiLink } from "@mui/material"
import type { SxProps } from "@mui/system"
import { Link as RouterLink } from "@tanstack/react-router"

import { headerColor } from "@/theme"

interface AppHeaderProps {
  sx?: SxProps
}

export default function AppHeader({ sx }: AppHeaderProps) {
  const menuContent = (<></>)

  return (
    <AppBar
      position="static"
      sx={{
        ...sx,
        height: "4rem",
        display: "flex",
        flexDirection: "row",
        justifyContent: "space-between",
        alignItems: "center",
        bgcolor: headerColor,
        boxShadow: "none",
      }}
    >
      <Box sx={{ ml: "1.5rem" }}>
        <MuiLink
          component={RouterLink}
          to="/"
          sx={{
            textDecoration: "none",
            color: colors.grey[300],
            fontSize: "1.75rem",
            letterSpacing: "0.25rem",
          }}
        >
          {"bsllmner-mk2"}
        </MuiLink>
      </Box>
      <Box sx={{ mr: "1.5rem" }}>{menuContent}</Box>
    </AppBar>
  )
}
