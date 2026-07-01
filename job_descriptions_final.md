# Tech Role Job Descriptions — AI Chatbot Prompts

> **Placeholders to fill before use:**
> - `[COMPANY_NAME]` — Name of the hiring company
> - `[COMPANY_STAGE]` — e.g., "Series A", "Seed", "Series B", "growth-stage"
> - `[COMPANY_DOMAIN]` — What the company does, e.g., "AI-native talent intelligence platform"
> - `[LOCATION_PRIMARY]` — Primary office city
> - `[LOCATION_SECONDARY]` — Secondary office city (remove if only one location)
> - `[TEAM_SIZE_CURRENT]` — Current engineering team size
> - `[TEAM_SIZE_TARGET]` — Target team size in 12 months
> - `[PRODUCT_NAME]` — Core product or platform name

---

## 0. Senior AI Engineer — Founding Team

**Company:** Redrob AI (Series A AI-native talent intelligence platform)
**Location:** Pune/Noida, India (Hybrid — flexible cadence) | Open to relocation candidates from Tier-1 Indian cities
**Employment Type:** Full-time
**Experience Required:** 5–9 years

### Let's be honest about this role

We're going to write this JD differently from most. We're a Series A company that just raised our round and we're building a new AI Engineering org from scratch. This is the kind of role where the JD changes every six months because the company changes every six months. So instead of pretending we have a fixed checklist, we're going to tell you what we actually need and what we've gotten wrong before.

If you've spent your career at Google or Meta and you want a well-scoped role with a defined ladder, this isn't it.

If you've spent your career bouncing between early-stage startups and you want to "just code" without having to think about product or recruiter workflows or eval frameworks, this also isn't it.

We need someone who is **simultaneously** comfortable with two things that sound contradictory:

Deep technical depth in modern ML systems — embeddings, retrieval, ranking, LLMs, fine-tuning.

Scrappy product-engineering attitude — willing to ship a working ranker in a week even if the underlying ML is "obviously suboptimal," because we need to learn from real users before we know what to actually optimize for.

These are not contradictory in real life. They feel contradictory because of how engineering culture sorted itself into "researcher" vs "shipper" archetypes. We need both modes available in the same person, and we'd rather you tilt slightly toward shipper than toward researcher.

### What you'd actually be doing

The high-level mandate: **own the intelligence layer of Redrob's product.** That means the ranking, retrieval, and matching systems that decide what recruiters see when they search for candidates and what candidates see when they search for roles.

In practical terms, your first 90 days will probably look like:

Weeks 1–3: Audit what we currently have (it's mostly BM25 + rule-based scoring, working but not great). Identify the 3–4 highest-leverage things to fix.

Weeks 4–8: Ship a v2 ranking system that demonstrably improves recruiter-engagement metrics. This will involve embeddings, hybrid retrieval, and probably some LLM-based re-ranking, but the architecture is your call.

Weeks 9–12: Set up the evaluation infrastructure — offline benchmarks, online A/B testing, recruiter-feedback loops — so we can keep improving without flying blind.

Beyond that, you'll be driving the long-term architecture of how we do candidate-JD matching at scale, mentoring the next round of hires (we're growing the team from 4 to 12 engineers in the next year), and working closely with our recruiter-experience PM on what to build.

### What we mean by "5–9 years"

This is a range, not a requirement. Some people hit "senior engineer" judgment at 4 years; some never hit it after 15. We've used 5–9 because it's roughly where people we've hired into this kind of role have landed, but we'll seriously consider candidates outside the band if other signals are strong.

**Disqualifiers we actually apply:**

If you've spent your career in pure research environments (academic labs, research-only roles) without any production deployment — we will not move forward. We've tried it twice and it didn't work for either side.

If your "AI experience" consists primarily of recent (under 12 months) projects using LangChain to call OpenAI — we will probably not move forward, unless you can demonstrate substantial pre-LLM-era ML production experience. We're looking for people who understood retrieval and ranking *before* it became fashionable.

If you are a senior engineer who hasn't written production code in the last 18 months because you've moved into "architecture" or "tech lead" roles — we will probably not move forward. This role writes code.

### The skills inventory

**Things you absolutely need:**

Production experience with **embeddings-based retrieval systems** (sentence-transformers, OpenAI embeddings, BGE, E5, or similar) deployed to real users. We don't care which model — we care that you've handled embedding drift, index refresh, retrieval-quality regression in production.

Production experience with **vector databases or hybrid search infrastructure** — Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, Elasticsearch, FAISS, or something similar.

Strong **Python**. Yes really, we care about code quality.

Hands-on experience designing **evaluation frameworks for ranking systems** — NDCG, MRR, MAP, offline-to-online correlation, A/B test interpretation. If you've never thought about how to evaluate a ranking system rigorously, this role will be very painful.

**Things we'd like but won't reject you for:**

LLM fine-tuning experience (LoRA, QLoRA, PEFT)
Experience with learning-to-rank models (XGBoost-based or neural)
Prior exposure to HR-tech, recruiting tech, or marketplace products
Background in distributed systems or large-scale inference optimization
Open-source contributions in the AI/ML space

**Things we explicitly do NOT want:**

**Title-chasers.** If your career trajectory shows you optimizing for "Senior" → "Staff" → "Principal" titles by switching companies every 1.5 years, we're not a fit.

**Framework enthusiasts.** If your GitHub is full of LangChain tutorials and your blog posts are "How I used [hot framework] to build [demo]" — that's fine but it's not what we need. We need people who think about systems, not frameworks.

**People who have only worked at consulting firms** (TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini, etc.) in their entire career.

**People whose primary expertise is computer vision, speech, or robotics** without significant NLP/IR exposure.

**People whose work has been entirely on closed-source proprietary systems for 5+ years** without external validation (papers, talks, open-source).

### On location and logistics

Pune/Noida-preferred but flexible. Offices in Noida and Pune (mostly used Tue/Thu). No fixed in-office day requirement but quarterly offsites expected. Candidates in Hyderabad, Pune, Mumbai, Delhi NCR welcome. Outside India: case-by-case, no visa sponsorship.

Notice period: sub-30-day preferred. Buyout up to 30 days available.

### The vibe check

We work async-first and write a lot. If you find writing painful, you'll find this role painful.

We disagree openly and decide quickly. If you find that style abrasive, you'll find this role abrasive.

We move fast and break things, with the caveat that "things" are usually our internal assumptions, not user-facing systems.

### How to read between the lines

The ideal candidate: 6–8 years total experience, of which 4–5 are in applied ML/AI roles at product companies (not pure services). Has shipped at least one end-to-end ranking, search, or recommendation system to real users at meaningful scale. Has strong opinions about retrieval, evaluation, and LLM integration — and can defend them with reference to systems they actually built.

---

## 1. Senior AI / ML Engineer — Founding Team

**Company:** [COMPANY_NAME] ([COMPANY_STAGE] [COMPANY_DOMAIN])
**Location:** [LOCATION_PRIMARY] / [LOCATION_SECONDARY], India (Hybrid — flexible cadence) | Open to relocation candidates from Tier-1 Indian cities
**Employment Type:** Full-time
**Experience Required:** 5–9 years

### Let's be honest about this role

We're going to write this JD differently from most. We're a [COMPANY_STAGE] company and we're building a new AI Engineering org from scratch. This is the kind of role where the JD changes every six months because the company changes every six months. So instead of pretending we have a fixed checklist, we're going to tell you what we actually need and what we've gotten wrong before.

If you've spent your career at Google or Meta and you want a well-scoped role with a defined ladder, this isn't it.

If you've spent your career bouncing between early-stage startups and you want to "just code" without having to think about product workflows or eval frameworks, this also isn't it.

We need someone who is **simultaneously** comfortable with two things that sound contradictory:

Deep technical depth in modern ML systems — embeddings, retrieval, ranking, LLMs, fine-tuning.

Scrappy product-engineering attitude — willing to ship a working ranker in a week even if the underlying ML is "obviously suboptimal," because we need to learn from real users before we know what to actually optimize for.

These are not contradictory in real life. They feel contradictory because of how engineering culture sorted itself into "researcher" vs "shipper" archetypes. We need both modes available in the same person, and we'd rather you tilt slightly toward shipper than toward researcher.

### What you'd actually be doing

The high-level mandate: **own the intelligence layer of [PRODUCT_NAME].** That means the ranking, retrieval, and matching systems that decide what users see when they search and what gets surfaced when the system reasons on their behalf.

In practical terms, your first 90 days will probably look like:

Weeks 1–3: Audit what we currently have (it's mostly rule-based scoring, working but not great). Identify the 3–4 highest-leverage things to fix.

Weeks 4–8: Ship a v2 ranking/retrieval system that demonstrably improves core engagement metrics. This will involve embeddings, hybrid retrieval, and probably some LLM-based re-ranking, but the architecture is your call.

Weeks 9–12: Set up the evaluation infrastructure — offline benchmarks, online A/B testing, user-feedback loops — so we can keep improving without flying blind.

Beyond that, you'll be driving the long-term architecture of how we do ML at scale, mentoring the next round of hires (we're growing the team from [TEAM_SIZE_CURRENT] to [TEAM_SIZE_TARGET] engineers in the next year), and working closely with product on what to build.

### What we mean by "5–9 years"

This is a range, not a requirement. Some people hit "senior engineer" judgment at 4 years; some never hit it after 15. We've used 5–9 because it's roughly where people we've hired into this kind of role have landed, but we'll seriously consider candidates outside the band if other signals are strong.

**Disqualifiers we actually apply:**

If you've spent your career in pure research environments (academic labs, research-only roles) without any production deployment — we will not move forward. We've tried it and it didn't work for either side.

If your "AI experience" consists primarily of recent (under 12 months) projects using LangChain to call OpenAI — we will probably not move forward, unless you can demonstrate substantial pre-LLM-era ML production experience.

If you are a senior engineer who hasn't written production code in the last 18 months because you've moved into "architecture" roles — we will probably not move forward. This role writes code.

### The skills inventory

**Things you absolutely need:**

Production experience with **embeddings-based retrieval systems** (sentence-transformers, OpenAI embeddings, BGE, E5, or similar) deployed to real users. We care that you've handled embedding drift, index refresh, and retrieval-quality regression in production.

Production experience with **vector databases or hybrid search infrastructure** — Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, Elasticsearch, FAISS, or similar.

Strong **Python**. We care about code quality.

Hands-on experience designing **evaluation frameworks for ranking systems** — NDCG, MRR, MAP, offline-to-online correlation, A/B test interpretation.

**Things we'd like but won't reject you for:**

LLM fine-tuning experience (LoRA, QLoRA, PEFT)
Experience with learning-to-rank models (XGBoost-based or neural)
Background in distributed systems or large-scale inference optimization
Open-source contributions in the AI/ML space

**Things we explicitly do NOT want:**

Title-chasers optimizing for "Senior" → "Staff" → "Principal" by switching companies every 1.5 years.

Framework enthusiasts whose GitHub is full of tutorials and demos without evidence of systems thinking.

People whose primary expertise is computer vision, speech, or robotics without significant NLP/IR exposure.

People whose work has been entirely on closed-source proprietary systems for 5+ years without external validation.

### The vibe check

We work async-first and write a lot. If you find writing painful, you'll find this role painful.

We disagree openly and decide quickly. If you find that style abrasive, you'll find this role abrasive.

We move fast. If you need a stable, mature codebase to be productive, this role will feel unstable.

---

## 2. Senior Backend Engineer

**Company:** [COMPANY_NAME] ([COMPANY_STAGE] [COMPANY_DOMAIN])
**Location:** [LOCATION_PRIMARY] / [LOCATION_SECONDARY], India (Hybrid) | Tier-1 city candidates welcome
**Employment Type:** Full-time
**Experience Required:** 4–8 years

### Let's be honest about this role

Most backend JDs look the same: a list of microservices, a checklist of databases, and some vague requirement about "scalability." We're going to be more specific.

We need someone who can build systems that hold up under real load — not demo load, not load-test load, but the kind of unpredictable production load where everything interesting happens at once. We've had engineers who were great at building things and engineers who were great at operating things. We need someone who's decent at both.

If you've never been woken up by a pager alert at 2am, debugged a production issue from logs alone, or had to explain to a PM why a perfectly-written service was still slow — you probably haven't seen enough of what this job actually involves.

### What you'd actually be doing

You'll own the API and data layer of [PRODUCT_NAME]. That means:

Designing and maintaining REST/gRPC APIs that our frontend and AI/ML layer depend on.

Owning the data model — schema design, migrations, and the performance implications of both.

Building the infra for async processing: job queues, event-driven workflows, background jobs.

Making the system observable: structured logging, metrics, distributed tracing — not as an afterthought, but as a first-class deliverable.

Reviewing code and setting engineering standards for a team that will grow from [TEAM_SIZE_CURRENT] to [TEAM_SIZE_TARGET] over the next year.

### The skills inventory

**Things you absolutely need:**

Production Python or Go (we use Python; Go is transferable). Not "I know the syntax" — production systems under real traffic.

Strong SQL — query design, index strategy, query plan analysis. If "EXPLAIN ANALYZE" means nothing to you, this will be painful.

Experience with at least one message queue or event streaming system (Kafka, RabbitMQ, Celery, SQS, Pub/Sub). Not configuration tutorials — actual async workflow design.

Proven ability to debug production issues: reading traces, reading query plans, reading flame graphs. The actual debugging, not just the theory of debugging.

REST API design — not just "I can write endpoints" but understanding of idempotency, versioning, pagination, and error contract design.

**Things we'd like but won't reject you for:**

Experience with Redis beyond caching (pub/sub, Lua scripting, sorted sets for rate limiting)
Familiarity with GraphQL or gRPC
Background in multi-tenant SaaS systems
Experience with PostgreSQL at meaningful scale (10M+ rows, non-trivial query optimization)

**Things we explicitly do NOT want:**

Engineers who treat observability as someone else's job. If you've never set up your own logging and metrics for a service you built, that's a red flag.

Engineers who think "add a cache" is always the answer to performance problems. Sometimes it is. Usually it's a symptom of a missing index.

Architects who've stopped writing code. This role writes code. A lot of it.

### The vibe check

We operate with small teams and high autonomy. If you need someone to spec every endpoint for you, this isn't the right environment.

We prefer boring technology that works over exciting technology that's new. If you want to introduce a new database every quarter, we're not a fit.

We ship, then improve. If you find it hard to ship something you're not 100% proud of, you'll find this role frustrating.

---

## 3. Senior Frontend Engineer

**Company:** [COMPANY_NAME] ([COMPANY_STAGE] [COMPANY_DOMAIN])
**Location:** [LOCATION_PRIMARY] / [LOCATION_SECONDARY], India (Hybrid) | Tier-1 city candidates welcome
**Employment Type:** Full-time
**Experience Required:** 3–7 years

### Let's be honest about this role

Most frontend JDs list React, TypeScript, and CSS and call it a day. That's not useful to you or to us.

Here's the honest version: we are building a complex, data-dense application where the UI is not decoration — it's the product. Users spend hours per day inside [PRODUCT_NAME]. If the interface is slow, confusing, or brittle, that is a product-level failure, and it will be partially yours to own.

We've hired frontend engineers who were great at building components but had no opinion about how those components composed into workflows. We don't need that. We need someone who looks at a user journey and thinks "this is going to be painful at scale" before we've built it.

### What you'd actually be doing

Building the core product UI in React with TypeScript. Not greenfield from scratch — you'll be working with an existing codebase that has real users and real technical debt.

Owning frontend performance. Not as a separate initiative — as a part of every PR. Bundle size, render performance, and perceived performance all matter here.

Designing and maintaining our component system. We don't want a new design system every time a new engineer joins — we want one that grows deliberately.

Working directly with designers and PMs to translate wireframes into real interfaces — and pushing back when a design will be slow, inaccessible, or unmaintainable.

Mentoring 1–2 junior engineers who will join as the team grows.

### The skills inventory

**Things you absolutely need:**

Production React (3+ years). We don't care if you learned it on hooks from day one or migrated from class components. We care that you understand the mental model — state, side effects, the component lifecycle, memoization tradeoffs.

TypeScript with strictness actually enabled. Not "we added TypeScript but most things are `any`." Real TypeScript.

CSS that actually works across viewports and browsers. Not Tailwind expertise — understanding of the cascade, specificity, layout models, and how they interact. Tailwind is fine; understanding what Tailwind does under the hood is required.

A real opinion about state management — when to use local state, when to use context, when to reach for a library. "We'll just use Redux for everything" is not an opinion.

Experience debugging frontend performance issues: profiling with DevTools, identifying unnecessary re-renders, diagnosing bundle bloat.

**Things we'd like but won't reject you for:**

Next.js (SSR, ISR, app router) experience
Familiarity with WebSocket or real-time UI patterns
Experience with data visualization (charts, tables with large datasets)
Accessibility experience (ARIA, keyboard navigation, WCAG compliance)

**Things we explicitly do NOT want:**

Engineers who conflate "good UI" with "complex animations." We need functional, fast, and clear over flashy.

Engineers who avoid touching CSS and delegate everything to a designer. If you can't debug a layout issue, you'll slow down every sprint.

Engineers who haven't worked in a shared codebase with other engineers. "I built this myself" is not the same as "I built this in a team."

### The vibe check

We care about craft but not perfectionism. Ship it, learn, improve. If you spend two weeks polishing a component that blocks other work, that's a problem.

Design and engineering work closely here. If "that's the designer's problem" is your instinct, you'll be frustrated.

We move fast. If you need complete specs before you can start building, you'll slow us down.

---

## 4. Full Stack Engineer

**Company:** [COMPANY_NAME] ([COMPANY_STAGE] [COMPANY_DOMAIN])
**Location:** [LOCATION_PRIMARY] / [LOCATION_SECONDARY], India (Hybrid) | Tier-1 city candidates welcome
**Employment Type:** Full-time
**Experience Required:** 3–6 years

### Let's be honest about this role

"Full stack" is one of the most abused labels in job descriptions. Sometimes it means "frontend engineer who occasionally touches a REST endpoint." Sometimes it means "backend engineer who once used React." We're going to define what we actually mean.

We mean: you can build a feature end-to-end. You can design a database schema, write the API, build the UI, write the tests, and ship it — with enough quality that you're not creating a maintenance problem for the next engineer. That's the scope.

We don't expect you to be equally good at everything. We expect you to be good enough across the stack that you're not blocked waiting for someone else on a task you could reasonably own.

### What you'd actually be doing

Building product features across the entire stack — from schema migrations to polished UI — with ownership of the full delivery.

Making architectural decisions at the feature level: which data model, which API shape, which frontend pattern. Not just following a spec.

Writing code others can maintain. Not "clever" code — readable code with clear intent and tests that actually test behavior.

Participating in on-call rotation and owning the features you ship in production, not just in staging.

Working closely with product and design to shape what you're building, not just execute on finalized specs.

### The skills inventory

**Things you absolutely need:**

A primary backend language in production — Python, Node.js, Go, or Ruby. You should be able to write an API that handles edge cases correctly without someone reviewing every line.

React with TypeScript in production. Not "I've done tutorials." Working product shipped to real users.

SQL proficiency — not just "I can write SELECT queries," but real understanding of indexing, joins, and what makes a query slow.

An understanding of how web applications actually work under the hood: HTTP, auth flows (JWT, sessions, OAuth), CORS, caching headers. Not just framework magic.

Experience writing tests — unit, integration, and at least some experience with end-to-end testing.

**Things we'd like but won't reject you for:**

Experience with containerization (Docker, Kubernetes)
CI/CD pipeline ownership
Familiarity with cloud services (AWS, GCP, or Azure)
Experience with job queues or event-driven systems

**Things we explicitly do NOT want:**

Engineers who identify exclusively as "full stack" but have never been the primary owner of a backend or frontend system of meaningful complexity.

Engineers who treat testing as optional or something to do "when there's time."

Engineers who need complete task specifications and don't ask questions during scoping. You'll be the one who catches missing edge cases before they become bugs.

### The vibe check

Small team, high autonomy. You'll own features, not tasks.

We prefer you saying "I don't know but I'll figure it out" over confidently guessing. Honest uncertainty is better than confident misinformation.

We move fast and we fix things. If you want a perfect codebase before you can be productive, this will be uncomfortable.

---

## 5. DevOps / Platform Engineer

**Company:** [COMPANY_NAME] ([COMPANY_STAGE] [COMPANY_DOMAIN])
**Location:** [LOCATION_PRIMARY] / [LOCATION_SECONDARY], India (Hybrid) | Tier-1 city candidates welcome
**Employment Type:** Full-time
**Experience Required:** 4–8 years

### Let's be honest about this role

Platform engineering roles get described as "infrastructure" when they're actually much broader. Let's be clear about what this role is: you are building and maintaining the platform that every other engineer at [COMPANY_NAME] depends on to do their work.

That means you are simultaneously a reliability engineer, a developer-experience engineer, and a security engineer. You won't be equally good at all three. That's fine. But you need to care about all three.

What we've gotten wrong before: hiring "DevOps engineers" who knew how to configure cloud services but didn't understand why applications were slow or unreliable. Configuration isn't operations. Operations is understanding what's happening at runtime and knowing what to do about it.

### What you'd actually be doing

Owning our CI/CD pipeline end-to-end — not just maintaining it, but improving the developer experience around it. Build times, test reliability, deployment safety.

Designing and operating our cloud infrastructure using infrastructure-as-code. Not clicking around in the console — reproducible, version-controlled infrastructure.

Owning observability: making sure every service is instrumented, every alert is actionable, and every incident has a paper trail.

Managing containerized workloads — deployments, autoscaling, resource allocation, and cost.

Being the person other engineers come to when they can't figure out why their service is behaving the way it is in production. That means you need to understand the applications, not just the infrastructure.

### The skills inventory

**Things you absolutely need:**

Production Kubernetes experience — not just deploying to a cluster, but understanding deployments, services, ingress, resource limits, HPA, and what happens when things go wrong.

Infrastructure-as-code proficiency — Terraform or Pulumi. Not "I've written some Terraform" — real production modules with state management and CI integration.

Cloud platform depth in at least one major provider (AWS, GCP, or Azure). Actual production experience with compute, networking, storage, and IAM at the level where you can debug unexpected charges and unexpected behavior.

Observability stack ownership — setting up and maintaining Prometheus/Grafana, or ELK, or equivalent. Understanding what metrics matter and what to alert on.

Scripting proficiency — Bash and Python, enough to automate operational tasks without creating unmaintainable scripts.

**Things we'd like but won't reject you for:**

Experience with service mesh (Istio, Linkerd)
Database operations experience (backups, replication, failover)
Security tooling experience (secret management, vulnerability scanning, RBAC)
Experience with cost optimization in cloud environments

**Things we explicitly do NOT want:**

Engineers who treat "it works in my environment" as an acceptable answer.

Engineers who build complex infrastructure without documenting it. If only you understand how it works, it's a liability.

Engineers who've only worked in environments with a dedicated SRE team handling incidents — we need people who have owned incidents directly.

### The vibe check

Platform engineers here are not a support function. You're a force multiplier. If you don't care about developer experience, you'll find the role frustrating.

Incidents happen. How you behave during and after an incident — calm, systematic, blameless — matters more than preventing every incident.

We run lean. If you need a team of 5 to manage what you'd own here, we're probably not the right fit.

---

## 6. Data Engineer

**Company:** [COMPANY_NAME] ([COMPANY_STAGE] [COMPANY_DOMAIN])
**Location:** [LOCATION_PRIMARY] / [LOCATION_SECONDARY], India (Hybrid) | Tier-1 city candidates welcome
**Employment Type:** Full-time
**Experience Required:** 3–7 years

### Let's be honest about this role

Data engineering is one of those disciplines where the job titles and the actual work have the widest gap in all of tech. "Data Engineer" at one company means ETL pipeline maintenance. At another it means building a full data platform from scratch. We're going to tell you which one this is.

This is closer to the second. Our data infrastructure is young, which means there's a lot to build — but also a lot of wrong turns we haven't taken yet. If you've only inherited mature data platforms and optimized them, you'll find the ambiguity here uncomfortable. If you've built things from scratch and have opinions about how to do it right, you'll find a lot of interesting problems.

### What you'd actually be doing

Designing and building our data pipelines from raw event data to analytics-ready tables. Ownership of the full pipeline, not just transformation logic.

Owning our data warehouse schema — not just loading data into it, but thinking about how it gets queried and making sure the model supports that.

Building the infrastructure that ML and AI features depend on — feature stores, training data pipelines, batch scoring pipelines.

Making data accessible to non-engineers: clean, documented tables that analysts and PMs can actually use without calling you every time.

Setting data quality standards and building the monitoring to enforce them. Not just "is the pipeline green?" but "is the data correct?"

### The skills inventory

**Things you absolutely need:**

Production Python for data pipelines — not notebooks, but production pipelines with proper error handling, retries, and logging.

SQL at the level of data modeling — not just writing queries, but designing schemas, understanding slowly changing dimensions, and knowing when a star schema is and isn't the right answer.

Experience with at least one orchestration tool in production (Airflow, Prefect, Dagster) — not just configuration, but building, testing, and debugging real DAGs.

Experience with a data warehouse (BigQuery, Snowflake, Redshift, or Databricks) at a scale where query performance and cost actually matter.

Data quality mindset — the ability to ask "how do I know this data is correct?" and actually answer it.

**Things we'd like but won't reject you for:**

Experience with streaming data pipelines (Kafka, Flink, Spark Streaming)
dbt for transformation modeling
Feature engineering for ML pipelines
Experience with data contracts or schema registries

**Things we explicitly do NOT want:**

Engineers who treat data quality as someone else's problem (usually "the source system's problem").

Engineers who've only worked with clean, well-structured data sources. Real data is messy. If you've never had to handle schema drift, late-arriving data, or upstream data corruption, you haven't seen enough.

Engineers who build pipelines that only they understand. Documentation is part of the job.

### The vibe check

Data engineering here sits close to the product and AI teams. You'll understand why the data matters, not just how to move it.

We'd rather have pipelines that are simple and reliable over pipelines that are technically impressive but fragile.

Data quality failures are a product problem. If you think of them as a data problem, you'll be siloed in a way that makes the role less effective.

---

## 7. Technical Product Manager

**Company:** [COMPANY_NAME] ([COMPANY_STAGE] [COMPANY_DOMAIN])
**Location:** [LOCATION_PRIMARY] / [LOCATION_SECONDARY], India (Hybrid) | Tier-1 city candidates welcome
**Employment Type:** Full-time
**Experience Required:** 3–7 years

### Let's be honest about this role

"Technical PM" is one of the most inflated labels in tech hiring. Most PMs who call themselves technical have a CS degree they haven't used in five years, can read code without being able to write it, and rely on engineers to validate technical feasibility.

That's not what we mean by technical PM. We mean: you can sit in a technical architecture discussion and contribute to it, not just observe it. You understand the difference between an API call and a database query in a way that matters for product decisions. You know when an engineer is overestimating complexity and when they're underestimating risk.

You don't need to write production code. But you need to have written enough code at some point that you understand the tradeoffs engineers make and can push back intelligently when you think they're wrong.

### What you'd actually be doing

Owning the roadmap for your product area — not just writing it down, but defending it in front of leadership, engineering, and design with actual reasoning about tradeoffs.

Writing product specs that engineers can actually work from. Not "a button that shows results" — specs with edge cases, error states, data models, and explicit out-of-scope decisions.

Working with AI/ML engineers to define evaluation criteria for features that involve models. If "make the AI smarter" is your idea of a success metric, this role isn't right for you.

Talking to users regularly and turning what they say into product decisions — not just feature requests but underlying problems.

Making tradeoff decisions between speed, quality, and scope that are defensible when they turn out wrong, not just when they turn out right.

### The skills inventory

**Things you absolutely need:**

Prior software engineering experience or a technical background where you've actually built things. We will ask you to walk us through something you built or contributed to technically.

Experience writing product specs with enough technical detail that engineers can estimate and implement without 10 clarification meetings.

Data literacy — ability to define metrics, query data to answer product questions, and tell the difference between correlation and causation in dashboards.

Comfort with ambiguity at the product level: ability to ship something with known limitations rather than waiting for the perfect version.

Stakeholder management with engineering — knowing when to push for something, when to defer to engineering judgment, and how to have the conversation when you disagree.

**Things we'd like but won't reject you for:**

Experience with AI/ML products — understanding of model evaluation, data pipelines, or inference latency
Background in the [COMPANY_DOMAIN] space
Experience at a high-growth startup where the PM:engineer ratio was low
Familiarity with analytics tools (Mixpanel, Amplitude, Metabase, Looker)

**Things we explicitly do NOT want:**

PMs who treat engineering as an execution function and product as a strategy function. We want collaborative product development, not a hand-off model.

PMs who write specs so vague that engineers make all the actual product decisions during implementation, then the PM claims credit for the outcome.

PMs who've never dealt with a feature going wrong in production and learned from it. If your experience is all greenfield launches, you haven't seen enough.

### The vibe check

Engineers here have high standards. If you can't defend your decisions with reasoning, you'll lose their respect quickly — and it's hard to earn back.

We move fast and we're wrong sometimes. If every failed experiment is a crisis, this role will be exhausting.

You'll have real ownership here. With small teams comes less cover when things go wrong, which is also true. If you need a lot of air cover, this isn't the right stage.

---

## 8. Mobile Engineer (iOS / Android)

**Company:** [COMPANY_NAME] ([COMPANY_STAGE] [COMPANY_DOMAIN])
**Location:** [LOCATION_PRIMARY] / [LOCATION_SECONDARY], India (Hybrid) | Tier-1 city candidates welcome
**Employment Type:** Full-time
**Experience Required:** 3–7 years

### Let's be honest about this role

Mobile development is one of the few engineering disciplines where the platform you're building for actively fights you. App Store reviews, OS update regressions, device fragmentation, battery and performance constraints that don't exist on the web — this is a harder environment to ship reliably in than most job descriptions acknowledge.

We need a mobile engineer who has been humbled by the platform and learned from it. Not someone who finds mobile interesting because of animations and gestures — those are fine, but they're not the job. The job is shipping a reliable, fast, well-behaved app that users trust.

### What you'd actually be doing

Building and maintaining the [PRODUCT_NAME] mobile app — features, bug fixes, and the infrastructure underlying both.

Owning app performance: startup time, scroll performance, memory usage, battery impact. These matter to real users even when they don't say so explicitly.

Managing the release pipeline — builds, signing, beta testing, app store submissions. Not delegating this — owning it.

Working closely with design to implement UI that matches intent while respecting platform conventions. Not just "pixel-perfect" — actually feeling native on each platform.

Handling the unglamorous parts: crash reporting, analytics instrumentation, deep link handling, push notifications, background tasks.

### The skills inventory

**Things you absolutely need:**

Proficiency in Swift (iOS) or Kotlin (Android) — or React Native with a genuine understanding of what's happening in the native layer, not just the JavaScript layer.

Experience shipping apps to the App Store or Google Play that real users downloaded and used. Not test builds — production apps.

Understanding of the mobile platform model: sandboxing, permissions, background execution limits, and why things that work in a web app don't translate directly.

Experience debugging crashes in production — reading symbolicated crash reports, understanding memory management issues, diagnosing ANRs.

API integration experience — not just calling endpoints, but handling network failures, offline states, and sync conflicts gracefully.

**Things we'd like but won't reject you for:**

Cross-platform experience (React Native, Flutter, or both native)
Experience with local storage and offline-first app patterns
Background in accessibility on mobile (Dynamic Type, VoiceOver/TalkBack)
Experience with performance profiling tools (Instruments, Android Profiler)

**Things we explicitly do NOT want:**

Mobile engineers who've only worked on internal enterprise apps with no performance constraints and no real users to disappoint.

Engineers who treat the mobile app as a thin client with no logic. State management on mobile is hard; pretending it isn't creates fragile apps.

Engineers whose answer to every performance problem is "upgrade the device" or "use a newer OS." Our users have older devices. We build for them.

### The vibe check

Mobile development requires patience with platform constraints. If you fight the platform instead of understanding it, you'll ship slow.

Our mobile app is a first-class product, not a feature. If you want to work on mobile but see web as the "real" engineering, this isn't the right fit.

We ship frequently. If you're used to quarterly release cycles and App Store approval is the only thing slowing you down, you'll find a faster cadence here disorienting in a good way.

---

*End of job descriptions. Replace all `[PLACEHOLDER]` values before deploying to the chatbot.*
