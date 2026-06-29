// Icons.jsx — small inline SVG icons shared across the design.
// All icons inherit currentColor and are sized via width/height props.

const Icon = ({ d, size = 16, fill = 'none', stroke = 'currentColor', sw = 1.8, viewBox = '0 0 24 24', children }) => (
  <svg width={size} height={size} viewBox={viewBox} fill={fill} stroke={stroke} strokeWidth={sw} strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    {d ? <path d={d} /> : children}
  </svg>
);

const I = {
  Shield: (p) => <Icon {...p} d="M12 3l8 3v6c0 4.5-3.2 8.4-8 9-4.8-.6-8-4.5-8-9V6l8-3z" />,
  Sparkles: (p) => <Icon {...p}><path d="M12 3v4M12 17v4M3 12h4M17 12h4M5.5 5.5l2.5 2.5M16 16l2.5 2.5M5.5 18.5L8 16M16 8l2.5-2.5"/></Icon>,
  ChevronRight: (p) => <Icon {...p} d="M9 6l6 6-6 6" />,
  ChevronDown: (p) => <Icon {...p} d="M6 9l6 6 6-6" />,
  Close: (p) => <Icon {...p} d="M6 6l12 12M18 6L6 18" />,
  Settings: (p) => <Icon {...p}><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.7 1.7 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.8-.3 1.7 1.7 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1a1.7 1.7 0 0 0-1-1.5 1.7 1.7 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0 .3-1.8 1.7 1.7 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1a1.7 1.7 0 0 0 1.5-1 1.7 1.7 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.8.3H9a1.7 1.7 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.8V9a1.7 1.7 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1z"/></Icon>,
  Check: (p) => <Icon {...p} d="M5 12l5 5L20 7" />,
  CheckCircle: (p) => <Icon {...p}><circle cx="12" cy="12" r="9"/><path d="M8 12.5l3 3 5-6"/></Icon>,
  AlertTriangle: (p) => <Icon {...p}><path d="M12 4l9.5 16h-19L12 4z"/><path d="M12 10v4M12 17.5v.1"/></Icon>,
  AlertOctagon: (p) => <Icon {...p}><path d="M8.5 3h7L21 8.5v7L15.5 21h-7L3 15.5v-7L8.5 3z"/><path d="M12 8v4M12 16v.1"/></Icon>,
  Info: (p) => <Icon {...p}><circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 8v.1"/></Icon>,
  ThumbsUp: (p) => <Icon {...p} d="M7 11v9H4a1 1 0 0 1-1-1v-7a1 1 0 0 1 1-1h3zm0 0l4-7a3 3 0 0 1 3 3v3h5a2 2 0 0 1 2 2.4l-1.4 7A2 2 0 0 1 16.6 20H7" />,
  ThumbsDown: (p) => <Icon {...p} d="M17 13V4h3a1 1 0 0 1 1 1v7a1 1 0 0 1-1 1h-3zm0 0l-4 7a3 3 0 0 1-3-3v-3H5a2 2 0 0 1-2-2.4l1.4-7A2 2 0 0 1 6.4 4H17" />,
  Star: (p) => <Icon {...p} fill="currentColor" stroke="none" d="M12 2l2.9 6.3 6.9.6-5.2 4.6 1.6 6.7L12 16.9 5.8 20.2l1.6-6.7L2.2 8.9l6.9-.6L12 2z" />,
  Lock: (p) => <Icon {...p}><rect x="5" y="11" width="14" height="9" rx="2"/><path d="M8 11V8a4 4 0 0 1 8 0v3"/></Icon>,
  ShoppingBag: (p) => <Icon {...p}><path d="M5 8h14l-1 12H6L5 8z"/><path d="M9 8V5a3 3 0 0 1 6 0v3"/></Icon>,
  // Metric icons
  ReviewAuth: (p) => <Icon {...p}><path d="M4 5h16v10H8l-4 4V5z"/><path d="M8.5 10h7"/><path d="M8.5 7h4"/></Icon>,
  Seller: (p) => <Icon {...p}><circle cx="12" cy="8" r="4"/><path d="M4 21a8 8 0 0 1 16 0"/></Icon>,
  Sentiment: (p) => <Icon {...p}><circle cx="12" cy="12" r="9"/><path d="M9 14s1 2 3 2 3-2 3-2"/><path d="M9 9.5v.1M15 9.5v.1"/></Icon>,
  Price: (p) => <Icon {...p}><path d="M3 12c0-5 4-9 9-9s9 4 9 9-4 9-9 9-9-4-9-9z"/><path d="M14.5 9.5C14 8.5 13 8 12 8c-1.5 0-2.5 1-2.5 2.2 0 2.6 5 1.6 5 4 0 1.4-1.2 2.3-2.5 2.3-1.2 0-2.4-.7-2.7-2"/><path d="M12 6.5V8M12 16v1.5"/></Icon>,
  Return: (p) => <Icon {...p}><path d="M3 12a9 9 0 1 0 3-6.7"/><path d="M3 4v5h5"/></Icon>,
  Brain: (p) => <Icon {...p}><path d="M9 4a3 3 0 0 0-3 3v.5A3 3 0 0 0 4 10v0a3 3 0 0 0 1 5.5V17a3 3 0 0 0 4 3"/><path d="M15 4a3 3 0 0 1 3 3v.5a3 3 0 0 1 2 2.5v0a3 3 0 0 1-1 5.5V17a3 3 0 0 1-4 3"/><path d="M12 4v16"/></Icon>,
  Search: (p) => <Icon {...p}><circle cx="11" cy="11" r="6"/><path d="M16 16l4 4"/></Icon>,
  Refresh: (p) => <Icon {...p}><path d="M3 12a9 9 0 0 1 15.5-6.3"/><path d="M21 4v5h-5"/><path d="M21 12a9 9 0 0 1-15.5 6.3"/><path d="M3 20v-5h5"/></Icon>,
  ExternalLink: (p) => <Icon {...p}><path d="M14 4h6v6"/><path d="M20 4l-9 9"/><path d="M19 14v5a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V6a1 1 0 0 1 1-1h5"/></Icon>,
  Database: (p) => <Icon {...p}><ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v6c0 1.7 3.6 3 8 3s8-1.3 8-3V5"/><path d="M4 11v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6"/></Icon>,
  Bolt: (p) => <Icon {...p} d="M13 2L3 14h7l-1 8 10-12h-7l1-8z" />,
};

window.I = I;
