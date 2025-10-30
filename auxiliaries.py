import calendar

def month_id_to_ym(month_id: int) -> str:
        """
        Converts month_id to month name and year

        Args:
            month_id: integer representing the month id
        
        Returns:
            String consisting of month name and year
        """

        offset = month_id - 1
        year = 1980 + offset // 12
        month_num = offset % 12 + 1
        month_name = calendar.month_name[month_num]

        return f"{month_name} {year}"

def get_id_column(
            loa: str
        ) -> str:
        """Get the ID column name based on the level of analysis (loa).

        Args:
            loa (str): Level of analysis; one of ["cm", "pgm"].

        Raises:
            ValueError: If loa is not "cm" or "pgm".

        Returns:
            str: The ID column name corresponding to the level of analysis.
        """

        if loa == "cm":
            return "country_id"
        elif loa == "pgm":
            return "pg_id"
        else:
            raise ValueError("loa must be 'cm' or 'pgm'")