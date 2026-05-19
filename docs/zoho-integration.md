# Zoho Integration

Penny's Zoho plugin provides unified access to Zoho Mail, Calendar, and Projects — enabling you to manage email, schedule events, and track tasks through natural conversation.

## Setup

### Required Environment Variables

```bash
ZOHO_API_ID=your_client_id
ZOHO_API_SECRET=your_client_secret
ZOHO_REFRESH_TOKEN=your_refresh_token
```

### Obtaining Credentials

1. Go to [Zoho API Console](https://api-console.zoho.com/)
2. Create a new **Self Client** application
3. Generate a refresh token with these scopes:
   - `ZohoMail.messages.ALL`
   - `ZohoMail.folders.ALL`
   - `ZohoCalendar.calendar.ALL`
   - `ZohoCalendar.event.ALL`
   - `ZohoProjects.portals.ALL`
   - `ZohoProjects.projects.ALL`
   - `ZohoProjects.tasks.ALL`

Use the following command in your terminal to generate a refresh token:

```bash
curl -X POST "https://accounts.zoho.com/oauth/v2/token" \
  -d "code=<AUTHORIZATION_CODE>" \
  -d "client_id=<ZOHO_API_ID>" \
  -d "client_secret=<ZOHO_API_SECRET>" \
  -d "grant_type=authorization_code"
```

---

## Email (`/email zoho`)

Search, read, organize, and draft email responses.

### Basic Usage

```text
/email zoho what packages am I expecting?
/email zoho any emails from mum this week?
/email zoho when is my dentist appointment?
```

### Drafting Replies

Penny can draft responses and save them to your Drafts folder for review:

```text
/email zoho draft a reply to John's email thanking him for the proposal
/email zoho respond to the AWS invoice email confirming receipt
```

> **Note:** Penny saves drafts for your review — she never sends emails directly.

### Email Organization

#### Moving Emails to Folders

Organize emails into nested folder structures:

```text
/email zoho move the AWS invoice to Accounting/Expenses/AWS
/email zoho file John's email under Clients/John Smith
/email zoho move all Stripe emails to Accounting/Payments
```

Folders are created automatically if they don't exist.

#### Applying Labels

Categorize emails without moving them:

```text
/email zoho label the invoice email as "completed"
/email zoho mark John's email as "pending"
/email zoho apply the "urgent" label to the support ticket
```

### Persistent Email Rules

Create rules that automatically organize incoming emails during scheduled checks:

```text
/email zoho create a rule to move all emails from AWS to Accounting/Expenses/AWS
/email zoho create a rule to label emails containing "invoice" as "accounting"
/email zoho list my email rules
```

#### Rule Conditions

- **from**: Match sender email or domain (`aws`, `@stripe.com`)
- **subject_contains**: Match text in subject line
- **body_contains**: Match text in email body

#### Rule Actions

- **move_to**: Move to folder path (`Accounting/Expenses/AWS`)
- **label**: Apply a label (`completed`, `urgent`)

#### Combining with Scheduled Checks

Rules are applied automatically when you schedule email checks:

```text
/schedule every morning at 9am check my email and apply rules
```

---

## Calendar (`/calendar`)

Check availability, view events, and schedule appointments.

### Viewing Your Schedule

```text
/calendar what's on my schedule this week?
/calendar what do I have tomorrow?
/calendar show me events for next Monday
```

### Checking Availability

```text
/calendar am I free on Friday at 2pm?
/calendar is Studio A available on December 15th from 10am to 2pm?
/calendar find a 1 hour slot for a meeting next week
```

### Creating Events

```text
/calendar create a meeting with John on Monday at 10am
/calendar schedule a session in Studio A on December 2nd from 2pm to 6pm
/calendar book a call with the team on Friday at 3pm for 30 minutes
```

### Multiple Calendars

If you have multiple calendars (e.g., "Default", "Studio A", "Personal"), specify which one:

```text
/calendar what's on Studio A this week?
/calendar create a session on Studio A for December 5th
/calendar is Personal calendar free on Saturday?
```

The "Default" calendar is used when no calendar is specified.

---

## Projects (`/project`)

Create projects, manage tasks, and track progress.

### Listing Projects

```text
/project list all projects
/project what projects do I have?
/project show me the Website Redesign project
```

### Creating Projects

```text
/project create a new project called "Website Redesign"
/project create project "Q1 Marketing Campaign" starting January 1st
```

### Managing Tasks

#### Creating Tasks

```text
/project add a task "Design homepage" to Website Redesign
/project create task "Write copy" in Website Redesign with high priority
/project add "Review mockups" to Website Redesign due next Friday
```

#### Viewing Tasks

```text
/project what tasks are in Website Redesign?
/project show me high priority tasks
/project list tasks in the Q1 Marketing Campaign
```

#### Updating Tasks

```text
/project mark "Design homepage" as 50% complete
/project set "Write copy" priority to high
/project update "Review mockups" to 100% complete
```

### Task Lists (Milestones)

Organize tasks into logical groups:

```text
/project create a task list "Phase 1" in Website Redesign
/project add "Design homepage" to the "Phase 1" task list
/project show task lists in Website Redesign
```

---

## Tips for Maximum Value

### 1. Automate Email Organization

Set up rules once, and let Penny handle the rest:

```text
/email zoho create a rule to move emails from @stripe.com to Accounting/Payments
/email zoho create a rule to move emails from @aws.amazon.com to Accounting/Expenses/AWS
/email zoho create a rule to label emails containing "urgent" as "urgent"
/schedule daily at 8am check my email and apply rules
```

### 2. Morning Briefing

Get a daily summary of what's ahead:

```text
/schedule every weekday at 8am tell me what's on my calendar today and any urgent emails
```

### 3. Project Tracking

Keep tasks organized and track progress:

```text
/project what tasks are incomplete in Website Redesign?
/project mark all design tasks as complete
/project what's the status of Q1 Marketing Campaign?
```

### 4. Smart Scheduling

Let Penny find available times:

```text
/calendar find three 1-hour slots next week for a client meeting
/calendar when is Studio A free for a 4-hour session this month?
```

### 5. Cross-Feature Workflows

Combine email, calendar, and projects:

```text
/email zoho check for any meeting requests and add them to my calendar
/email zoho look for project updates and create tasks in Website Redesign
```

---

## Troubleshooting

### "No calendars found"

Ensure your refresh token includes the `ZohoCalendar.calendar.ALL` scope.

### "No projects found"

Verify you have at least one portal in Zoho Projects and the `ZohoProjects.portals.ALL` scope.

### "Failed to create folder"

Check that your Zoho Mail account has permission to create folders.

### Token Expired

If you see authentication errors, generate a new refresh token from the Zoho API Console.
