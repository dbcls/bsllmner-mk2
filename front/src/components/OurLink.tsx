import { OpenInNewOutlined } from "@mui/icons-material"
import { Link } from "@mui/material"
import { type SxProps } from "@mui/system"

interface OurLinkProps {
  sx?: SxProps
  href: string
  text: string
}

export default function OurLink({ sx, href, text }: OurLinkProps) {
  return (
    <Link
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      underline="hover"
      sx={{ display: "inline-flex", alignItems: "center", gap: "0.25rem", ...sx }}
    >
      {text}
      <OpenInNewOutlined sx={{ fontSize: "1rem" }} />
    </Link>
  )
}
