import { colors } from "@mui/material"
import { createTheme } from "@mui/material/styles"

export const theme = createTheme({
  palette: {
    primary: {
      main: colors.indigo[400],
    },
    info: {
      main: colors.grey[800],
    },
    warning: {
      main: colors.deepOrange[500],
    },
    text: {
      primary: colors.grey[900],
      secondary: colors.grey[600],
    },
    background: {
      default: colors.grey[100],
    },
  },
  breakpoints: {
    values: {
      xs: 0,
      sm: 600,
      md: 960,
      lg: 1280,
      xl: 1920,
    },
  },
})

export const headerColor = "#333D4D"

export default theme
