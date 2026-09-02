export interface TeamMember {
  name: string;
  initials: string;
  hue: number;          // avatar accent hue
  github: string;
  linkedin: string;     // the direct contact line for a VC
  role?: string;        // short strength line — only set once confirmed accurate
}

// LinkedIn is the primary contact path (VCs write directly to a person).
// GitHub stays as a secondary link. URLs are owner-supplied on the page.
export const team: TeamMember[] = [
  {
    name: 'Alexander Kuznetsov',
    initials: 'AK',
    hue: 45,
    github: 'https://github.com/AlexSKuznetsov',
    linkedin: 'https://www.linkedin.com/in/alexskuznetsov/',
    role: 'AI systems · real-time architecture · computer vision',
  },
  {
    name: 'Nadine Chernova',
    initials: 'NC',
    hue: 330,
    github: 'https://github.com/NadChern',
    linkedin: 'https://www.linkedin.com/in/nadinechern/',
    role: 'Product · wearable engineering',
  },
  {
    name: 'Erin Shih',
    initials: 'ES',
    hue: 145,
    github: 'https://github.com/erinshih413',
    linkedin: 'https://www.linkedin.com/in/erin-shih-06b46b16b/',
    role: 'Mobile app',
  },
  {
    name: 'Jacky Huang',
    initials: 'JH',
    hue: 220,
    github: 'https://github.com/jack980180',
    linkedin: 'https://www.linkedin.com/in/shao-yu-huang-14320b244/',
    role: 'STT/TTS · media server improvements',
  },
];
