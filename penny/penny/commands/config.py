"""The /config command — view and modify runtime configuration parameters."""

from __future__ import annotations

from penny.commands.base import Command
from penny.commands.models import CommandContext, CommandResult
from penny.config_params import RUNTIME_CONFIG_PARAMS


class ConfigCommand(Command):
    """View and modify runtime configuration parameters."""

    name = "config"
    description = "View and modify runtime configuration parameters"
    help_text = (
        "View and modify runtime configuration parameters like timing settings. "
        "Configuration is stored in the database and takes effect immediately.\n\n"
        "**Usage**:\n"
        "- `/config` — List all available configuration parameters and their current values\n"
        "- `/config <key>` — Show the value of a specific configuration parameter\n"
        "- `/config <key> <value>` — Update a configuration parameter"
    )

    async def execute(self, args: str, context: CommandContext) -> CommandResult:
        """Execute config command."""
        from datetime import UTC, datetime

        from sqlmodel import Session, select

        from penny.database.models import RuntimeConfig

        parts = args.strip().split(maxsplit=1)

        # Case 1: List all config
        if not args.strip():
            lines = ["**Runtime Configuration**", ""]

            # List all parameters with current values (from config object)
            for key in sorted(RUNTIME_CONFIG_PARAMS.keys()):
                param = RUNTIME_CONFIG_PARAMS[key]
                field_name = key.lower()
                current_value = getattr(context.config, field_name, param.default_value)
                lines.append(f"- **{key}**: {current_value} ({param.description})")

            lines.append("")
            lines.append("Use `/config <key> <value>` to change a setting.")
            return CommandResult(text="\n".join(lines))

        # Case 2: Get specific config
        if len(parts) == 1:
            key = parts[0].upper()
            if key not in RUNTIME_CONFIG_PARAMS:
                return CommandResult(
                    text=(
                        f"unknown config parameter: {key}\n"
                        "Use /config to see all available parameters."
                    )
                )

            param = RUNTIME_CONFIG_PARAMS[key]
            field_name = key.lower()
            current_value = getattr(context.config, field_name, param.default_value)
            return CommandResult(text=f"**{key}**: {current_value} ({param.description})")

        # Case 3: Set config value
        key = parts[0].upper()
        value_str = parts[1]

        if key not in RUNTIME_CONFIG_PARAMS:
            return CommandResult(
                text=(
                    f"unknown config parameter: {key}\nUse /config to see all available parameters."
                )
            )

        param = RUNTIME_CONFIG_PARAMS[key]

        # Validate value
        try:
            parsed_value = param.validator(value_str)
        except ValueError as e:
            return CommandResult(text=f"invalid value for {key}: {e}")

        # Store in database
        with Session(context.db.engine) as session:
            existing = session.exec(select(RuntimeConfig).where(RuntimeConfig.key == key)).first()

            if existing:
                existing.value = str(parsed_value)
                existing.updated_at = datetime.now(UTC)
                session.add(existing)
            else:
                new_config = RuntimeConfig(
                    key=key,
                    value=str(parsed_value),
                    description=param.description,
                    updated_at=datetime.now(UTC),
                )
                session.add(new_config)

            session.commit()

        # Config changes take effect immediately via __getattribute__
        return CommandResult(text=f"ok, updated {key} to {parsed_value}")
