"""personal object registry

Revision ID: 9f8c7b6a5d4e
Revises: 4aec0811a363
Create Date: 2026-08-15 12:30:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "9f8c7b6a5d4e"
down_revision: str | Sequence[str] | None = "4aec0811a363"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "registry_state",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "enrolled_objects",
        sa.Column("object_id", sa.String(length=64), nullable=False),
        sa.Column("label", sa.String(length=128), nullable=False),
        sa.Column("idempotency_key", sa.String(length=512), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("registry_version", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("object_id"),
    )
    with op.batch_alter_table("enrolled_objects", schema=None) as batch_op:
        batch_op.create_index(
            batch_op.f("ix_enrolled_objects_idempotency_key"),
            ["idempotency_key"],
            unique=True,
        )
        batch_op.create_index(batch_op.f("ix_enrolled_objects_label"), ["label"], unique=False)
        batch_op.create_index(
            batch_op.f("ix_enrolled_objects_registry_version"),
            ["registry_version"],
            unique=False,
        )

    op.create_table(
        "object_views",
        sa.Column("view_id", sa.String(length=64), nullable=False),
        sa.Column("object_id", sa.String(length=64), nullable=False),
        sa.Column("view_index", sa.Integer(), nullable=False),
        sa.Column("quality", sa.JSON(), nullable=False),
        sa.Column("embedder_id", sa.String(length=256), nullable=False),
        sa.Column("pooling", sa.String(length=128), nullable=False),
        sa.Column("dim", sa.Integer(), nullable=False),
        sa.Column("summary", sa.LargeBinary(), nullable=False),
        sa.Column("pooled_spatial", sa.LargeBinary(), nullable=False),
        sa.Column("crop_sha256", sa.String(length=64), nullable=False),
        sa.Column("crop_media_type", sa.String(length=64), nullable=False),
        sa.Column("crop_relative_path", sa.String(length=512), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("registry_version", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(["object_id"], ["enrolled_objects.object_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("view_id"),
        sa.UniqueConstraint(
            "object_id", "view_index", "crop_sha256", name="uq_object_view_content"
        ),
    )
    with op.batch_alter_table("object_views", schema=None) as batch_op:
        batch_op.create_index(
            batch_op.f("ix_object_views_crop_sha256"), ["crop_sha256"], unique=False
        )
        batch_op.create_index(
            "ix_object_views_object_index", ["object_id", "view_index"], unique=False
        )
        batch_op.create_index(batch_op.f("ix_object_views_object_id"), ["object_id"], unique=False)
        batch_op.create_index(
            batch_op.f("ix_object_views_registry_version"),
            ["registry_version"],
            unique=False,
        )


def downgrade() -> None:
    with op.batch_alter_table("object_views", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_object_views_registry_version"))
        batch_op.drop_index(batch_op.f("ix_object_views_object_id"))
        batch_op.drop_index("ix_object_views_object_index")
        batch_op.drop_index(batch_op.f("ix_object_views_crop_sha256"))
    op.drop_table("object_views")

    with op.batch_alter_table("enrolled_objects", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_enrolled_objects_registry_version"))
        batch_op.drop_index(batch_op.f("ix_enrolled_objects_label"))
        batch_op.drop_index(batch_op.f("ix_enrolled_objects_idempotency_key"))
    op.drop_table("enrolled_objects")
    op.drop_table("registry_state")
