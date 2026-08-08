"""Effective-looks computation for the statistical cost model's `nlooks`."""

__all__ = ["get_effective_looks"]


def get_effective_looks(
    range_looks: float,
    azimuth_looks: float,
    range_spacing: float,
    azimuth_spacing: float,
    range_resolution: float,
    azimuth_resolution: float,
) -> float:
    r"""
    Compute the equivalent number of independent looks for `cuphu.unwrap`'s
    ``nlooks`` argument.

    ``nlooks`` is **not** ``range_looks * azimuth_looks``: real SAR data is
    typically oversampled relative to its physical resolution, so adjacent
    pixels are correlated and the naive product overstates the number of
    truly independent samples going into the coherence estimate — often by
    an order of magnitude. This function applies the standard correction
    (the same formula used by ISCE3/NISAR and documented, uncomputed, in
    snaphu-py's own docstring):

    .. math:: n_e = k_r k_a \frac{d_r d_a}{\rho_r \rho_a}

    Parameters
    ----------
    range_looks, azimuth_looks : float
        Number of looks (multilook factor) in range and azimuth, i.e. the
        :math:`k_r`, :math:`k_a` used to form the multilooked interferogram.
    range_spacing, azimuth_spacing : float
        Single-look sample spacing in range and azimuth (:math:`d_r`,
        :math:`d_a`), in the same units as the resolution arguments.
    range_resolution, azimuth_resolution : float
        Range and azimuth resolution (:math:`\rho_r`, :math:`\rho_a`), in
        the same units as the spacing arguments. For SAR data this is
        typically ``c / (2 * range_bandwidth)`` for range and
        ``platform_velocity / azimuth_bandwidth`` for azimuth.

    Returns
    -------
    nlooks : float
        Effective number of independent looks, suitable for
        `cuphu.unwrap`'s ``nlooks`` argument.

    Examples
    --------
    Illustrative numbers only — spacing and resolution are NOT fixed
    per-mission constants (they depend on the acquisition/bandwidth mode
    of the specific product); read them from that product's own metadata
    (e.g. an RSLC's ``slantRangeSpacing``/``rangeBandwidth``/
    ``azimuthBandwidth`` plus its orbit for platform velocity), not from a
    hardcoded table. Units just need to match between the two spacing and
    the two resolution arguments:

    >>> get_effective_looks(
    ...     range_looks=13, azimuth_looks=16,
    ...     range_spacing=2.33, azimuth_spacing=2.75,      # m, single-look
    ...     range_resolution=7.5, azimuth_resolution=6.0,  # m
    ... )
    29.616888888888894

    Note this is far below the naive product ``13 * 16 = 208`` — the
    correction for oversampling matters.

    Notes
    -----
    This mirrors ``nisar.workflows.unwrap.get_effective_looks`` in ISCE3,
    but takes plain numeric spacing/resolution instead of RSLC/orbit
    objects — cuPHU has no radar-geometry awareness of its own, so callers
    with access to that metadata (e.g. ISCE3) should compute
    ``range_spacing``/``azimuth_spacing``/``range_resolution``/
    ``azimuth_resolution`` themselves and pass plain floats here.

    References
    ----------
    .. [1] C. W. Chen and H. A. Zebker, "Two-dimensional phase unwrapping
       with use of statistical models for cost functions in nonlinear
       optimization," J. Opt. Soc. Am. A, vol. 18, pp. 338-351 (2001).
    """
    return (
        range_looks
        * azimuth_looks
        * (range_spacing * azimuth_spacing)
        / (range_resolution * azimuth_resolution)
    )
