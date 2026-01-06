import { Box, Card, CardContent, SvgIconProps, Typography } from '@mui/material';

interface StatCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  icon: React.ReactElement<SvgIconProps>;
  color?: 'primary' | 'secondary' | 'success' | 'warning' | 'error' | 'info';
}

export default function StatCard({ title, value, subtitle, icon, color = 'primary' }: StatCardProps) {
  const colorMap = {
    primary: { bg: '#eff6ff', text: '#2563eb' },
    secondary: { bg: '#f5f3ff', text: '#7c3aed' },
    success: { bg: '#ecfdf5', text: '#10b981' },
    warning: { bg: '#fffbeb', text: '#f59e0b' },
    error: { bg: '#fef2f2', text: '#ef4444' },
    info: { bg: '#eff6ff', text: '#3b82f6' },
  };

  return (
    <Card className="card-hover">
      <CardContent sx={{ p: 3 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
          <Box>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 0.5 }}>
              {title}
            </Typography>
            <Typography variant="h4" sx={{ fontWeight: 700, mb: 0.5 }}>
              {value}
            </Typography>
            {subtitle && (
              <Typography variant="caption" color="text.secondary">
                {subtitle}
              </Typography>
            )}
          </Box>
          <Box
            sx={{
              p: 1.5,
              borderRadius: 2,
              bgcolor: colorMap[color].bg,
              color: colorMap[color].text,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            {icon}
          </Box>
        </Box>
      </CardContent>
    </Card>
  );
}
